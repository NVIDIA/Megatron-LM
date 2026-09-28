import importlib.util

# These tests type-check Megatron with the optional spmd_types package.
if importlib.util.find_spec("spmd_types") is None:
    collect_ignore_glob = ["test_*.py"]
