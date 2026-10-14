# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional framework compiler caches for the GDN output-fusion CI bucket."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import distributions
from pathlib import Path

GDN_BUCKET = "tests/unit_tests/ssm/test_gdn_gated_output_norm_fusion.py"
CACHE_DIR = "/opt/megatron-lm/assets_dir/compiler-cache/gdn"


def compiler_cache_env(
    scope: str, test_case: str, environment: str, tag: str | None
) -> dict[str, str]:
    """Keep compiler artifacts in the repository mount without affecting test selection."""
    if (scope, test_case, environment, tag) != ("unit-tests", GDN_BUCKET, "dev", "latest"):
        return {}
    return {
        "TORCHINDUCTOR_CACHE_DIR": f"{CACHE_DIR}/inductor",
        "TRITON_CACHE_DIR": f"{CACHE_DIR}/triton",
        "TILELANG_CACHE_DIR": f"{CACHE_DIR}/tilelang",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
    }


def cache_prefix(root: Path, recipe_platform: str, runtime: dict) -> str:
    """Separate incompatible runtimes; frameworks still validate individual kernel keys."""
    paths = [root / "pyproject.toml", root / "uv.lock"]
    paths.extend(path for path in (root / "docker").rglob("*") if path.is_file())
    identity = {
        "platform": recipe_platform,
        "runtime": runtime,
        "build_inputs": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)
        },
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return f"unit-compiler-gdn-v1-main-{recipe_platform}-{digest}-"


def runtime_identity() -> dict:
    """Read the compiler environment inside the same image and GPU runtime as the tests."""
    import torch

    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "packages": sorted(
            (distribution.metadata["Name"], distribution.version)
            for distribution in distributions()
        ),
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "gpus": sorted(
            {
                (torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
                for index in range(torch.cuda.device_count())
            }
        ),
    }


def main() -> None:
    """Print the cache namespace for the Actions restore and save steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True)
    args = parser.parse_args()
    prefix = cache_prefix(Path(__file__).resolve().parents[3], args.platform, runtime_identity())
    print(f"cache_prefix={prefix}")


if __name__ == "__main__":
    main()
