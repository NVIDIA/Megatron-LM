## 2024-05-20 - [PyTorch Training Loop Optimization]
**Learning:** Extracting inner functions defined inside hot loops (like training steps or FLOPs calculation) to the module level significantly reduces function creation overhead, especially in high-frequency calls.
**Action:** Always check frequently called functions for inner function definitions that don't rely on closure over changing variables.
## 2025-10-27 - Test Environment Restrictions
**Learning:** Unit tests try to download data to `/opt/data` which is read-only in sandbox.
**Action:** Use `UNIT_TEST_DATA_DIR` env var or patch `conftest.py` to redirect to a writable path like `/tmp/data`.

## 2025-10-27 - CPU Testing
**Learning:** The codebase defaults to NCCL and CUDA for distributed init, causing tests to fail on CPU-only environments.
**Action:** Patch `tests/unit_tests/test_utilities.py` to use `gloo` backend if CUDA is unavailable.

## 2025-10-27 - PyTorch Optimization
**Learning:** Using `torch.tensor(list_of_tensors)` or `float(tensor)` causes significant CPU-GPU synchronization overhead.
**Action:** Use `torch.stack` to combine tensors and keep computations on the device. Avoid converting tensors to Python scalars in hot paths.

## 2025-10-27 - Tensor Reduction Optimization
**Learning:** Using `max(t.abs().max() for t in tensors)` triggers a CPU-GPU synchronization for every tensor in the list because Python's `max` needs to compare values.
**Action:** Use `torch.stack([t.abs().max() for t in tensors]).max()` to keep all operations on the GPU and reduce synchronization overhead to a single call.
## 2025-10-27 - torch._foreach_norm Precision Safety
**Learning:** `torch._foreach_norm` accumulates in the input dtype and lacks a `dtype` argument. Using it on FP16/BF16 inputs can cause overflow if the sum of squares exceeds the type's range (e.g., > 65504 for FP16).
**Action:** Restrict `torch._foreach_norm` optimization to `torch.float32` inputs. For lower precision, fallback to `torch.norm(..., dtype=torch.float32)` or manually cast before calling.
