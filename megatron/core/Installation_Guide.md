# Installing Megatron Core

Megatron Core maintains a lightweight installation and minimizes conflicts by keeping its core dependencies (torch, numpy, and packaging) to a minimum. This is achieved through "import-guarding," where additional dependencies are only verified and loaded when the specific features that require them are actively used.

There are two ways of extending Megatron Core with its requirements that unlock the performance required for large-scale distributed training: Using a NGC PyTorch container or installing from source. While the installation into a NGC PyTorch container may simplify experience by shipping with pre-installed performance optimized dependencies, a source installation gives more freedom and customization options. In the following sections, we will have a look at both.

Before we dive into the fully-featured installation process, let's have a quick detour to the basic installation process.

## Basic installation

Megatron Core ships released wheels to PyPi bi-monthly.

```shell
pip install megatron-core
```

Additionally, there are weekly pre-release wheels:

```shell
pip install --pre megatron-core
```

Specific commits can be installed from the official NVIDIA/Megatron-LM GitHub repository:

```shell
pip install git+https://github.com/NVIDIA/Megatron-LM.git@${COMMIT}
```

Each installation method has complete feature-parity for a selected version.

## Installation inside a NGC PyTorch container

The [NGC PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) container includes NVIDIA system-level dependencies such as NCCL, CUDA, and cuDNN, which provide lower-level GPU support. It also comes with Python libraries specifically optimized and compiled for these software versions. Two key libraries for Megatron Core are a performance-optimized version of PyTorch, which incorporates advanced performance features not yet available in upstream [Meta PyTorch](https://github.com/pytorch/pytorch) at the time of release, and [NVIDIA Transformer Engine](<https://github.com/NVIDIA/TransformerEngine/>).

To get started, run the following commands:

```bash
# On the host machine
docker run --rm -it --gpus all nvcr.io/nvidia/pytorch:XX.YY-py3

# Inside the container
pip install megatron-core
```

:bulb: For the most recently tested NGC PyTorch image visit the file `.gitlab/stages/01.build.yml`. The stable release branches are named `core_rX.Y.Z`.

For a complete installation of Megatron Core with all features, follow these steps:

```bash
# Inside the container
pip install -U setuptools packaging
pip install --no-build-isolation megatron-core[dev]
```

:bulb: We add the argument `--no-build-isolation` since many dependencies like `transformer-engine` need to be aligned with the pre-installed CUDA and torch version. By removing Python's default build isolation, we expose the installation process to the host and its software versions. As a result, the compiler is able to build the source specific to those versions.

This command also installs libraries such as [flash-infer](https://github.com/flashinfer-ai/flashinfer), [mamba-ssm](https://github.com/state-spaces/mamba), and [grouped-gemm](https://github.com/fanshiqing/grouped_gemm). Depending on your CUDA and PyTorch environment versions, the installation could take anywhere from a few seconds to over thirty minutes.

This situation arises because most dependencies offer a wide array of pre-compiled wheels, compatible with various combinations of CUDA, PyTorch, and their respective library versions. When a suitable pre-compiled wheel is located, installation is nearly instantaneous. Conversely, if no such wheel exists, the local host machine must compile the source-distributed wheel into a binary. Feel free to raise an issue at NVIDIA/Megatron-LM if you identify such an issue and we will check if we can accelerate the installation of your use case.

The dev extra-requires option includes all dependencies validated by Megatron-LM's internal CI. This may be more extensive than necessary for your specific needs. You can review the requirements file at <https://github.com/NVIDIA/Megatron-LM/blob/main/pyproject.toml#L68-L86> to select only the dependencies relevant to your use case.

```bash
# Inside the container
# Example to only install support for hybrid models
pip install --no-build-isolation \
  megatron-core \
  "mamba-ssm~=2.2" \
  "causal-conv1d~=1.5" \
  "nv-grouped-gemm~=1.1"
```

## Installation inside a vanilla Ubuntu container

While pre-configured NGC PyTorch containers are often suitable, some use cases may necessitate a custom container. Other noteworthy NGC containers include [NGC cuda](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda) or [NGC cuda-dl-base](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda-dl-base).

For educational purposes, the following section details a "bare-metal" installation within a plain Ubuntu environment. This demonstration aims to provide users with sufficient knowledge to manage installations within pre-configured NGC containers.

### Preliminary requirements

The software stack used for this guide was configured with Ubuntu 24.04, Cuda 12.8, cuDNN 9.1, Python 3.12, PyTorch 2.8, and Transformer Engine 2.5.0.

#### Starting the container

```bash
docker run --rm -it --entrypoint bash ubuntu:24.04
```

### Installing Python

We will install Python 3.12 development headers, Python 3.12 `venv` for virtual environment support, and `pip` for installing additional packages. For convenience, `update-alternatives` will be used to set `python` as the default command instead of `python3.12`.

```shell
apt-get update
apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt-get install -y python3.12-dev python3.12-venv python3-pip 
update-alternatives --install /usr/bin/python python /usr/bin/python3 1
```

### Installing Cuda-toolkit

To establish a clean CUDA development environment on Ubuntu 24.04, we begin by installing essential tools such as `wget`, `curl`, `git`, and `cmake` for software downloading and building. We then remove any existing CUDA/NVIDIA repositories to prevent conflicts. Subsequently, NVIDIA's official CUDA keyring is retrieved and installed, which securely integrates the latest CUDA repository into the system. The final step involves installing the CUDA Toolkit 12.8 (comprising the compiler, runtime, and libraries), cuDNN 9 (for GPU-accelerated deep learning primitives), and CUTLASS (a template library for high-performance matrix operations).

```shell
# Install tools
apt-get update
apt-get install -y wget curl git cmake

rm /etc/apt/sources.list.d/cuda*.list || true
rm /etc/apt/sources.list.d/nvidia-cuda.list || true

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

apt-get update
apt-get install -y cuda-toolkit-12-8 \
libcudnn9-cuda-12 \
libcutlass-dev 
```

### Python libraries

Finally, we can set up a virtual Python environment and run a feature-complete installation of Megatron Core:

```shell
python -m venv .venv
source .venv/bin/activate
# Run this first to install basic dependencies and build-requirements for step two
pip install megatron-core
# Run this for the feature-complete install
pip install --no-build-isolation megatron-core[dev]
```

## Testing correctness

After successful installation of Megatron Core and its dependencies, we can validate the environment by the following commands.

For testing Megatron Core,  the following command should be successful:

```python
import megatron.core

print(megatron.core.__version__) 
```

For testing Transformer Engine, the following command should be successful:

```python
import transformer_engine
import transformer_engine.pytorch

print(transformer_engine.__version__)
```

## Summary

This guide has aimed to facilitate the installation and operational understanding of `Megatron Core`, including its continuous integration and deployment mechanisms. We trust this resource will prove valuable in the seamless development of large-scale LLMs using Megatron Core. Your insights and feedback are highly valued as we continue to enhance this tool. We encourage you to share your experiences, report any issues, or propose improvements by engaging with our GitHub community at [github.com/NVIDIA/Megatron-LM/issues](http://github.com/NVIDIA/Megatron-LM/issues). Your contributions are instrumental in shaping the future development of Megatron Core.
