# SM100 fused GDR backward

This package provides the fused CuTe DSL backward kernel used by the Megatron
Core internal Gated Delta Rule backend on Blackwell SM100 GPUs.

## Recommended usage

Select the internal backend through `TransformerConfig`:

```python
config = TransformerConfig(..., gdn_gdr_backend="internal")
```

`MCORE_GDN_INTERNAL_BACKEND` controls runtime dispatch:

- `auto` (default) uses the fused kernel when its input contract is satisfied
  and otherwise falls back to FLA.
- `cute` requires the CuTe DSL path and reports unsupported inputs as errors.
- `fla` bypasses the CuTe DSL path.

The backend adapter prepares the forward state and calls `fused_gdr_bwd`
internally. Model code should normally use the configured GDN layer instead of
calling the low-level wrapper directly.

## Input contract

The low-level wrapper accepts exactly two packed sequences:

- SM100 CUDA device and contiguous tensors on the same device.
- `q`, `k`, `v`, and `do`: BF16 `[1, N, 64, 128]`.
- `a`: BF16 `[1, N, 64, 64]`.
- `g` and `beta`: FP32 `[1, N, 64]`.
- `h`: BF16 `[1, N / 64, 64, 128, 128]`.
- `dht`: FP32 `[2, 64, 128, 128]`.
- `cu_seqlens`: contiguous CUDA int32 `[0, L0, L0 + L1]`, where both
  sequence lengths are positive multiples of 64 and `N = L0 + L1`.
- `chunk_size` must be 64, `state_v_first` must be `False`, and `scale` must
  be finite and positive.

The return tuple is `(dq, dk, dv, dg, dbeta, dh0)` with shapes and dtypes
matching the corresponding inputs.

## Notes

- Dense `B=2` inputs are packed by the Megatron adapter before launch.
- The adapter promotes gate and beta inputs to FP32 and restores their gradient
  dtypes after the kernel returns.
- When no final-state gradient is requested, the adapter supplies a cached zero
  `dht` tensor.
- Packed-sequence metadata is cached by `cu_seqlens` identity and tensor
  version; mutating the tensor invalidates the cached metadata.
- Unsupported shapes and dtypes fall back only in `auto` mode.
