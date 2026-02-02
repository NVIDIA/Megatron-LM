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

## 2025-10-27 - Multi-Tensor Operations
**Learning:** `torch._foreach_norm` provides a significant speedup (e.g. 2.4x on CPU) over iterating with `torch.norm` for list of tensors, but requires careful handling of return types (list/tuple of tensors) and dtype casting.
**Action:** Prefer `_foreach` variants for multi-tensor operations, guarding with `hasattr(torch, '_foreach_...')` for compatibility.
