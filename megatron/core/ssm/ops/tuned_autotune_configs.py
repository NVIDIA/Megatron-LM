# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pre-tuned Triton autotune winners used by deterministic mode.

Deterministic mode cannot let Triton pick a kernel config by benchmarking: the
winner depends on wall-clock timings, so two identical runs can select different
tile shapes and produce different floating-point reduction orders. Selecting the
cheapest config instead is deterministic but gives up throughput, because the
cheapest config is rarely the fastest one.

This table closes that gap. It records the config the autotuner actually chose,
measured once offline per GPU architecture, so deterministic mode can pin the
*fastest* config while remaining a pure function of its inputs.

Regenerate with a normal (autotuning) run of the target workload::

    MCORE_DET_TUNE_RECORD=/path/to/records torchrun ... pretrain.py ...
    python -m megatron.core.ssm.ops.tuned_autotune_configs /path/to/records.*.json

Layout::

    TUNED_AUTOTUNE_CONFIGS[arch][kernel_name][key] = {
        "kwargs": {...}, "num_warps": int, "num_stages": int
    }

``arch`` is ``sm<major><minor>``. ``key`` is the shape/dtype signature built by
``_tuning_key`` in ``determinism.py``; the literal ``"*"`` matches any shape and
is useful for kernels whose best config does not vary with the traced shapes.

A missing entry is not an error: deterministic mode falls back to the min-cost
config and warns. It never falls back to timing-based selection.
"""

TUNED_AUTOTUNE_CONFIGS: dict[str, dict[str, dict[str, dict]]] = {
    # Populated by a recording run; see the module docstring.
}


def _merge_records(paths):
    """Merge per-rank recording files into a single table."""
    import json

    merged: dict[str, dict[str, dict[str, dict]]] = {}
    for path in paths:
        with open(path) as handle:
            for arch, kernels in json.load(handle).items():
                for kernel, entries in kernels.items():
                    merged.setdefault(arch, {}).setdefault(kernel, {}).update(entries)
    return merged


def _render(table) -> str:
    """Render a merged table as the literal body of this module's dict."""
    import pprint

    return (
        "TUNED_AUTOTUNE_CONFIGS: dict[str, dict[str, dict[str, dict]]] = "
        + pprint.pformat(table, indent=4, width=100, sort_dicts=True)
        + "\n"
    )


if __name__ == "__main__":
    import sys

    print(_render(_merge_records(sys.argv[1:])))
