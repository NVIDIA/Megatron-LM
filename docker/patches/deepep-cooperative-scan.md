# HybridEP metadata scan cooperative launch

[`deepep-cooperative-scan.patch`](deepep-cooperative-scan.patch) vendors
[DeepEP PR #792](https://github.com/deepseek-ai/DeepEP/pull/792) at commit
[`cdf824fa593a7df35067c28a774f3804e3d44727`](https://github.com/Autumn1998/DeepEP/commit/cdf824fa593a7df35067c28a774f3804e3d44727).
The patch retains the upstream author, sign-off and MIT license notice.

The metadata scan polls partial sums produced by other CUDA thread blocks.
An ordinary launch can hang when resident blocks wait for blocks that have not
been scheduled. The patch uses `cudaLaunchKernelEx` with
`cudaLaunchAttributeCooperative` so the grid must fit a cooperative launch.
An unsupported launch or oversized grid fails through `CUDA_CHECK`; there is no
fallback to the ordinary launch. The scan computation and its arguments stay
unchanged.

`docker/Dockerfile.ci.dev` applies the patch before compiling DeepEP, after the
existing setup and NVSHMEM patches. It retains the DeepEP pin
`10d4dd7377d5bce900fbb4b80cce863764892b95` and the existing build options.
The LTS image is outside the scope of this backport.

For a separate DeepEP source build, apply the patch before compiling the
extension. The same patch also applies to the public `hybrid-ep` revision
`58397eac493b075b0baea5861ac6e9793d8d7a21`:

```bash
# Run from the Megatron-LM repository root.
PATCH="$(pwd)/docker/patches/deepep-cooperative-scan.patch"
git clone https://github.com/deepseek-ai/DeepEP.git
git -C DeepEP checkout 58397eac493b075b0baea5861ac6e9793d8d7a21
git -C DeepEP apply --check "$PATCH"
git -C DeepEP apply "$PATCH"
```

Rebuild the extension using the CUDA, PyTorch, transport and GPU architecture
settings for your environment. Applying the source patch does not change an
already installed DeepEP binary. Runtime validation must cover the supported
GPU and dispatch configurations, including permute fusion where enabled.

When updating the DeepEP pin to a revision containing this upstream fix, remove
the backport and its Dockerfile application together.
