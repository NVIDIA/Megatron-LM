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

    # TRITON_CACHE_AUTOTUNING=1 is required here: with it unset, an installed
    # mamba_ssm that ships its own import-time pin has already reduced every
    # config list to one entry, so there is no benchmark left to record.
    MCORE_DET_TUNE_RECORD=/path/to/records TRITON_CACHE_AUTOTUNING=1 torchrun ... pretrain.py ...
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

# Recorded from 96-GPU Nemotron-3-Ultra runs, majority vote over the ranks.
#   sm100 (GB200): 15 kernels, 16 entries; 6 entries had ranks disagreeing.
#   sm103 (GB300): 15 kernels, 16 entries; 4 entries had ranks disagreeing.
# That disagreement is the variance this table removes.
TUNED_AUTOTUNE_CONFIGS: dict[str, dict[str, dict[str, dict]]] = {
    'sm100': {
        '_bmm_chunk_bwd_kernel': {
            'chunk_size=128|K=128|torch.bfloat16|torch.float32|torch.bfloat16|torch.float32': {
                'kwargs': {'BLOCK_SIZE_CS': 32, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128},
                'num_stages': 4,
                'num_warps': 4,
            }
        },
        '_bmm_chunk_fwd_kernel': {
            'chunk_size=128|K=128|IS_CAUSAL=False|torch.bfloat16|torch.bfloat16|torch.float32': {
                'kwargs': {'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64},
                'num_stages': 4,
                'num_warps': 4,
            }
        },
        '_chunk_cumsum_bwd_kernel': {
            'chunk_size=128|nheads=128|torch.float32|torch.float32|torch.bfloat16|torch.float32|torch.float32|torch.bfloat16|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_H': 8},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_chunk_cumsum_fwd_kernel': {
            'chunk_size=128|nheads=128|torch.bfloat16|torch.float32|torch.float32|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_H': 8},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_chunk_scan_bwd_dc_kernel': {
            'chunk_size=128|dstate=128|hdim=64|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_chunk_scan_bwd_dcb_kernel': {
            'chunk_size=128|hdim=64|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_chunk_scan_bwd_ddAcs_stable_kernel': {
            'chunk_size=128|hdim=64|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_chunk_scan_bwd_dstates_kernel': {
            'hdim=64|dstate=128|chunk_size=128|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64},
                'num_stages': 4,
                'num_warps': 2,
            }
        },
        '_chunk_scan_bwd_dz_kernel': {
            'chunk_size=128|hdim=64|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.float32': {
                'kwargs': {'BLOCK_SIZE_M': 64},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_chunk_scan_chunk_state_bwd_dx_kernel': {
            'chunk_size=128|hdim=64|dstate=128|torch.bfloat16|torch.float32|torch.bfloat16|torch.float32|torch.float32|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64},
                'num_stages': 4,
                'num_warps': 4,
            }
        },
        '_chunk_scan_fwd_kernel': {
            'chunk_size=128|hdim=64|dstate=128|IS_CAUSAL=True|torch.float32|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32|torch.bfloat16|torch.bfloat16|torch.bfloat16': {
                'kwargs': {'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64},
                'num_stages': 5,
                'num_warps': 2,
            }
        },
        '_chunk_state_bwd_db_kernel': {
            'chunk_size=128|dstate=128|hdim=64|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_chunk_state_fwd_kernel': {
            'hdim=64|dstate=128|chunk_size=128|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64},
                'num_stages': 4,
                'num_warps': 2,
            }
        },
        '_state_passing_bwd_kernel': {
            'dim=8192|torch.float32|torch.float32|torch.float32|torch.bfloat16|torch.float32|torch.bfloat16': {
                'kwargs': {'BLOCK_SIZE': 512},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_state_passing_fwd_kernel': {
            'dim=8192|torch.float32|torch.bfloat16|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE': 512},
                'num_stages': 3,
                'num_warps': 4,
            },
            'dim=8192|torch.float32|torch.float32|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE': 512},
                'num_stages': 3,
                'num_warps': 4,
            },
        },
    },
    'sm103': {
        '_bmm_chunk_bwd_kernel': {
            'chunk_size=128|K=128|torch.bfloat16|torch.float32|torch.bfloat16|torch.float32': {
                'kwargs': {'BLOCK_SIZE_CS': 32, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128},
                'num_stages': 4,
                'num_warps': 4,
            }
        },
        '_bmm_chunk_fwd_kernel': {
            'chunk_size=128|K=128|IS_CAUSAL=False|torch.bfloat16|torch.bfloat16|torch.float32': {
                'kwargs': {'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128},
                'num_stages': 4,
                'num_warps': 4,
            }
        },
        '_chunk_cumsum_bwd_kernel': {
            'chunk_size=128|nheads=128|torch.float32|torch.float32|torch.bfloat16|torch.float32|torch.float32|torch.bfloat16|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_H': 16},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_chunk_cumsum_fwd_kernel': {
            'chunk_size=128|nheads=128|torch.bfloat16|torch.float32|torch.float32|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_H': 8},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_chunk_scan_bwd_dc_kernel': {
            'chunk_size=128|dstate=128|hdim=64|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_chunk_scan_bwd_dcb_kernel': {
            'chunk_size=128|hdim=64|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_chunk_scan_bwd_ddAcs_stable_kernel': {
            'chunk_size=128|hdim=64|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_chunk_scan_bwd_dstates_kernel': {
            'hdim=64|dstate=128|chunk_size=128|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64},
                'num_stages': 4,
                'num_warps': 2,
            }
        },
        '_chunk_scan_bwd_dz_kernel': {
            'chunk_size=128|hdim=64|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.float32': {
                'kwargs': {'BLOCK_SIZE_M': 64},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_chunk_scan_chunk_state_bwd_dx_kernel': {
            'chunk_size=128|hdim=64|dstate=128|torch.bfloat16|torch.float32|torch.bfloat16|torch.float32|torch.float32|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64},
                'num_stages': 4,
                'num_warps': 4,
            }
        },
        '_chunk_scan_fwd_kernel': {
            'chunk_size=128|hdim=64|dstate=128|IS_CAUSAL=True|torch.float32|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32|torch.bfloat16|torch.bfloat16|torch.bfloat16': {
                'kwargs': {'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64},
                'num_stages': 5,
                'num_warps': 2,
            }
        },
        '_chunk_state_bwd_db_kernel': {
            'chunk_size=128|dstate=128|hdim=64|torch.bfloat16|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_chunk_state_fwd_kernel': {
            'hdim=64|dstate=128|chunk_size=128|torch.bfloat16|torch.bfloat16|torch.float32|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64},
                'num_stages': 4,
                'num_warps': 2,
            }
        },
        '_state_passing_bwd_kernel': {
            'dim=8192|torch.float32|torch.float32|torch.float32|torch.bfloat16|torch.float32|torch.bfloat16': {
                'kwargs': {'BLOCK_SIZE': 512},
                'num_stages': 3,
                'num_warps': 4,
            }
        },
        '_state_passing_fwd_kernel': {
            'dim=8192|torch.float32|torch.bfloat16|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE': 512},
                'num_stages': 3,
                'num_warps': 4,
            },
            'dim=8192|torch.float32|torch.float32|torch.float32|torch.float32': {
                'kwargs': {'BLOCK_SIZE': 512},
                'num_stages': 3,
                'num_warps': 4,
            },
        },
    },
}


def _merge_records(paths):
    """Merge per-rank recording files by majority vote.

    Ranks routinely disagree about the winner - that disagreement is the very
    non-determinism this table exists to remove - so the merge counts votes
    rather than letting the last file win. Ties break on the serialized config
    so the result is reproducible for a given set of recordings.
    """
    import collections
    import json

    votes: dict = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    )
    for path in sorted(paths):
        with open(path) as handle:
            for arch, kernels in json.load(handle).items():
                for kernel, entries in kernels.items():
                    for key, config in entries.items():
                        votes[arch][kernel][key][json.dumps(config, sort_keys=True)] += 1

    merged: dict[str, dict[str, dict[str, dict]]] = {}
    for arch in sorted(votes):
        for kernel in sorted(votes[arch]):
            for key in sorted(votes[arch][kernel]):
                counter = votes[arch][kernel][key]
                winner = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                merged.setdefault(arch, {}).setdefault(kernel, {})[key] = json.loads(winner)
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
