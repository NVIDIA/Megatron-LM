## 2025-10-27 - Test Environment Restrictions
**Learning:** Unit tests try to download data to `/opt/data` which is read-only in sandbox.
**Action:** Use `UNIT_TEST_DATA_DIR` env var or patch `conftest.py` to redirect to a writable path like `/tmp/data`.

## 2025-10-27 - CPU Testing
**Learning:** The codebase defaults to NCCL and CUDA for distributed init, causing tests to fail on CPU-only environments.
**Action:** Patch `tests/unit_tests/test_utilities.py` to use `gloo` backend if CUDA is unavailable.
