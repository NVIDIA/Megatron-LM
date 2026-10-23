# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Expose locked FA4/CuTeDSL packages alongside the NGC image's compiled FA2."""

import importlib.metadata
import importlib.util
import shutil
import site
import sysconfig
from pathlib import Path


def main():
    """Repair only the package paths shared with the system-site-packages base image."""
    venv_site = Path(sysconfig.get_path("purelib")).resolve()
    distributions = {
        dist.metadata["Name"].lower().replace("_", "-"): dist
        for dist in importlib.metadata.distributions(path=[str(venv_site)])
    }
    fa4 = distributions["flash-attn-4"]
    source = Path(fa4.locate_file("flash_attn/cute")).resolve()
    if not source.is_relative_to(venv_site) or not (source / "interface.py").is_file():
        raise RuntimeError(f"Missing locked FlashAttention 4 package: {source}")

    fa2_spec = importlib.util.find_spec("flash_attn")
    if fa2_spec is None or fa2_spec.origin is None:
        raise RuntimeError("The base image must provide the compiled FlashAttention 2 package")
    fa2_package = Path(fa2_spec.origin).resolve().parent
    if not (fa2_package / "flash_attn_interface.py").is_file():
        raise RuntimeError(f"Unexpected FlashAttention 2 package: {fa2_package}")
    target = fa2_package / "cute"
    if target != source:
        if target.is_symlink():
            raise RuntimeError(f"Refusing to replace a symlinked FA2 subpackage: {target}")
        # Replace this one subpackage, including stale files from FA2's bundled CuTe code.
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    print(f"FlashAttention 4 {fa4.version}: {source} -> {target}")

    # Older system CuTeDSL wheels insert their own dsl_packages ahead of the venv.
    # Disable only their recorded path hooks, after verifying the locked replacement.
    cutlass = distributions["nvidia-cutlass-dsl"]
    hooks = [entry for entry in cutlass.files or () if str(entry).endswith(".pth")]
    if not hooks or not all(Path(cutlass.locate_file(entry)).is_file() for entry in hooks):
        raise RuntimeError("The locked CuTeDSL distribution must provide its path hook")
    for location in site.getsitepackages():
        system_site = Path(location).resolve()
        if system_site == venv_site:
            continue
        for dist in importlib.metadata.distributions(path=[str(system_site)]):
            name = dist.metadata["Name"].lower().replace("_", "-")
            if name != "nvidia-cutlass-dsl":
                continue
            for entry in dist.files or ():
                hook = Path(dist.locate_file(entry))
                if hook.suffix != ".pth" or not hook.name.startswith("nvidia_cutlass_dsl"):
                    continue
                if not hook.is_file():
                    continue
                if hook.resolve().parent != system_site or "dsl_packages" not in hook.read_text():
                    raise RuntimeError(f"Unexpected system CuTeDSL path hook: {hook}")
                hook.rename(hook.with_suffix(".pth.disabled-by-mcore"))
                print(f"Disabled system CuTeDSL {dist.version} path hook: {hook}")


if __name__ == "__main__":
    main()
