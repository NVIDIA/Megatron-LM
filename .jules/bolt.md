## 2024-05-20 - [PyTorch Training Loop Optimization]
**Learning:** Extracting inner functions defined inside hot loops (like training steps or FLOPs calculation) to the module level significantly reduces function creation overhead, especially in high-frequency calls.
**Action:** Always check frequently called functions for inner function definitions that don't rely on closure over changing variables.
## 2025-10-27 - Test Environment Restrictions
**Learning:** Unit tests try to download data to `/opt/data` which is read-only in sandbox.
**Action:** Use `UNIT_TEST_DATA_DIR` env var or patch `conftest.py` to redirect to a writable path like `/tmp/data`.

## 2025-10-27 - CPU Testing
**Learning:** The codebase defaults to NCCL and CUDA for distributed init, causing tests to fail on CPU-only environments.
**Action:** Patch `tests/unit_tests/test_utilities.py` to use `gloo` backend if CUDA is unavailable.

## 2025-10-27 - [Avoid torch.tensor on list of tensors]
**Learning:** `torch.tensor(list_of_tensors)` creates a new tensor by copying data, often involving host synchronization or inefficient copying, especially if the list contains CUDA tensors.
**Action:** Use `torch.stack(list_of_tensors)` to stack tensors efficiently on the same device.
