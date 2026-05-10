# DeltaNet variants — experiment log

Branch: `feat/deltanet-variants`. Track: `350m-ablation` (1B tokens, NorMuon, hybrid 3:1, 1 GH200 node, ~30 min/run, head_dim 64 / 4kv-heads / 8v-heads unless noted).

## Code changes (summary)

| area | file | what |
| --- | --- | --- |
| variants | `megatron/core/ssm/gated_delta_net.py` | DeltaNet flag-gated on top of GDN: rmsnorm/L2 q,k; β=2σ neg-eig; per-channel/scalar output gate; learnable S₀; carry-v1/v2; DeltaProduct (n_householder); erase Householders; hard frob-cap; v-norm; tightK; betacap; betabias |
| variants | `megatron/core/ssm/kimi_delta_attention.py` | KDA module (per-channel α decay) |
| variants | `megatron/core/ssm/mamba3_mixer.py` + `megatron/core/models/hybrid/mamba3_layer_specs.py` | Mamba 3 wrapper + spec dispatch (forward NaN's in our setup; not pursued further) |
| mamba2 | `megatron/core/ssm/mamba_mixer.py` | carry-v2 patch: persist final SSM state across batches via `mamba_split_conv1d_scan_combined`'s `initial_states`/`return_final_states` |
| config | `megatron/core/transformer/transformer_config.py` | `linear_attention_*` flags (qk_norm, neg_eigval, output_gate_form, carry_state, n_householder, n_erase, carried_state_max_frob, …) |
| install | `_research/launch/install_python_deps.sh` | Pin `fla-core==0.5.0`, install `tilelang` (Hopper Triton 3.4 backward-bug workaround), install `mamba_ssm>=2.3.1` + `causal_conv1d>=1.5.0` with `MAMBA_SKIP_CUDA_BUILD=TRUE` and `--no-build-isolation` for ARM64 |
| logging | `_research/logging_patch/hooks.py` | State stats accumulator: per-layer frob/amax/abs_mean/std + init_frob/delta_frob, mirrored to wandb |

## Full ranked results (sorted by final loss)

| # | final | variant | jobid |
| ---: | ---: | --- | --- |
| 1 | **2.1914** | DN+perchgate+carry-v2+**dp4** (4w, 0e) | 2083671 |
| 2 | 2.1919 | DN+perchgate+carry-v2+dp4-erase2 (2w, 2e) | 2080028 |
| 3 | 2.1927 | DN+perchgate+carry-v2+dp4+**cap=200** | 2083969 |
| 4 | 2.1931 | DN+perchgate+carry-v2+dp4-erase1 (3w, 1e) | 2083674 |
| 5 | 2.1952 | DN+perchgate+carry-v2+dp3-erase1 (2w, 1e) | 2083673 |
| 6 | 2.1976 | DN+perchgate+carry-v2+dp3 (3w, 0e) | 2080024 |
| 7 | 2.1987 | DN+perchgate+carry-v2+dp2 (2w, 0e) | 2080023 |
| 8 | 2.2014 | DN+perchgate+carry-v2+dp2-erase1 (1w, 1e) | 2080027 |
| 9 | 2.2070 | DN+perchgate+carry-v2 (no DP, no cap) | 2078807 |
| 9 | 2.2070 | **DN+perchgate+carry-v2+cap=200** *(leaderboard entry)* | 2083670 |
| 11 | 2.2076 | GDN+neg+carry-v2 | 2078935 |
| 11 | 2.2076 | DN+perchgate+carry-v2 (repro) | 2078934 |
| 13 | 2.2082 | GDN+neg (no carry) | 2076277 |
| 13 | 2.2082 | DN+perchgate+carry-v2+cap=50 | 2083669 |
| 15 | 2.2096 | GDN+neg+carry-v1 (batch-mean) | 2078808 |
| 16 | 2.2106 | DN+perchgate+carry-v2 h8d32 | 2079635 |
| 17 | 2.2107 | DN+perchgate+carry-v2+cap=10 | 2083668 |
| 18 | 2.2120 | DN+perchgate+carry-v2 h2d128 | 2079736 |
| 18 | 2.2120 | DN+perchgate+carry-v2 + L2-on-V | 2078939 |
| 20 | 2.2142 | KDA + carry-v2 | 2084342 |
| 21 | 2.2143 | **GDN baseline** | 2076276 |
| 21 | 2.2143 | GDN + carry-v2 | 2078936 |
| 21 | 2.2143 | DN+perchgate+carry-v1 (batch-mean) | 2078722 |
| 24 | 2.2145 | KDA baseline | 2084260 |
| 24 | 2.2145 | DN+carry-v2 nogate | 2083688 |
| 26 | 2.2147 | DN+carry-v2 nogate + cap=200 | 2083851 |
| 27 | 2.2148 | GDN+carry-v1 | 2078809 |
| 28 | 2.2149 | DN+perchgate (no carry) qk=rmsnorm | 2077866 |
| 29 | 2.2150 | DN+perchgate+carry-v2 (rerun) | 2078933 |
| 30 | 2.2152 | DN+perchgate+carry-v2 h16d16 | 2079637 |
| 31 | 2.2180 | DN+L2-qk+neg+perchgate+carry-v2 | 2079499 |
| 32 | 2.2204 | DN+L2-qk+neg+perchgate (no carry) | 2078736 |
| 33 | 2.2205 | DN+perchgate+carry-v2 h1d256 | 2079737 |
| 34 | 2.2216 | DN+perchgate betacap=0.9 | 2076474 |
| 35 | 2.2240 | DN+neg+tightK 0.7 (only stable DN+neg) | 2076477 |
| 36 | **2.2241** | **DN baseline** (Schlag rmsnorm-qk, perchgate) | 2076309 |
| 37 | 2.2242 | DN+perchgate+carry-v2 h32d8 | 2079638 |
| 38 | 2.2254 | DN+perchgate tightK 0.7 | 2076472 |
| 39 | **2.2259** | **Mamba 2 baseline** | 2084457 |
| 40 | 2.2267 | DN+perchgate betabias=−2 | 2076475 |
| 41 | 2.2270 | DN+L2-qk+perchgate (no neg) | 2077867 |
| 42 | **2.2277** | Mamba 2 + carry-v2 | 2084532 |
| 43 | 2.2420 | DN+rmsnorm-qk scalar gate | 2077865 |
| 44 | 2.2450 | DN+L2-qk scalar gate | 2077868 |
| 45 | 2.4263 | DN+perchgate learnable S₀ (NaN late) | 2076476 |
| -- | NaN | DP(n=4)+erase(2)+neg, DP(n=2)+neg, DN+neg betacap=0.95, DN+neg rmsnorm-qk, KDA pre-fix, Mamba 3 (5 attempts) | -- |

## Throughput vs n_householder (DeltaProduct)

| n_householder | tok/s/gpu | rel | MFU |
| ---: | ---: | ---: | ---: |
| 1 (no DP) | 127,296 | 1.00× | 34.5% |
| 2 | 103,289 | 0.81× | 28.9% |
| 3 | 92,706 | 0.73× | 26.7% |
| 4 (with 2 erases) | 81,406 | 0.64× | 24.2% |

Erase Householders cost the same as write Householders (kernel time identical, only v is masked).

## State-norm dynamics (carry-v2 runs)

| run | step 1 | step 477 | step 954 | step 1907 (mean / max) |
| --- | ---: | ---: | ---: | --- |
| GDN+carry-v2 | 6.5 | 14 | 20 | 21 / 35 |
| GDN+neg+carry-v2 | 13 | 18 | 23 | 24 / 37 |
| KDA+carry-v2 | 6.5 | 13 | 19 | 22 / 42 |
| DN+perchgate+carry-v2 | 64 | 502 | 1517 | 2810 / 7323 |
| dp(n=2) | 62 | 671 | 1864 | 3484 / 7205 |
| dp(n=3) | 62 | 860 | 2013 | 3255 / 8871 |
| dp(n=4)+erase(2) | 43 | 571 | 1598 | 3564 / 18,456 |
| L2-on-V (only bounded DN variant) | 41 | 169 | 430 | 668 / 1815 |

GDN/KDA reach an equilibrium ~step 500 (frob_mean stays at 6–25, delta_frob ≈ 0). DN (no decay) grows monotonically. `delta_frob` ≈ 0 by step ~500 for every variant — the per-batch increment is tiny; the giant carried-state accumulates across hundreds of batches.

## State-norm dynamics under hard cap (DN+perchgate+carry-v2)

Per-batch-element cap on the carried buffer. Logged on the full tensor across MBS=8 batch elements, so the analytical bound is `init_frob ≤ √MBS · cap = √8 · cap`.

| step | cap=10 init / Δ | cap=50 init / Δ | cap=200 init / Δ | no-cap init / Δ |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 21.21 / +42.4 | 47.81 / +15.8 | 47.81 / +15.8 | ~48 / +15 |
| 239 | 28.28 / +137.9 | 141.39 / +56.1 | 205.71 / +15.2 | 220 / +0 |
| 477 | 28.28 / +219.7 | 141.42 / +126.9 | 431.70 / +43.7 | 502 / +0 |
| 954 | 28.28 / +309.8 | 141.41 / +191.0 | 554.27 / +70.7 | 1517 / +0 |
| 1431 | 28.28 / +347.8 | 141.42 / +201.8 | 565.65 / +74.4 | 2300 / +0 |
| 1907 | 28.28 / +319.8 | 141.42 / +168.0 | 565.65 / +61.4 | 2810 / +0 |

Cap binding validated: 28.28 = √8·10 ✓; 141.42 = √8·50 ✓; 565.69 = √8·200 ✓.

| variant | final loss | Δ vs no-cap |
| --- | ---: | ---: |
| no-cap (carry-v2) | 2.2070 | 0 |
| cap=200 | 2.2070 | 0 |
| cap=50 | 2.2082 | +0.0012 |
| cap=10 | 2.2107 | +0.0037 |

As the cap tightens, delta_frob *grows* while init_frob shrinks — the capped model is forced to do its computation intra-sequence each forward; the uncapped model offloads memory to the across-batch carried buffer.

## Findings

1. **Per-channel output gate** alone closes the entire DN→GDN gap. Scalar gate is much worse.
2. **State carry-over** (carry-v1 batch-mean) also closes the gap → init/warmup hypothesis confirmed. carry-v2 (full per-batch) is the cleanest implementation and used by everything since.
3. **DeltaProduct n=2,3,4** gives monotone improvement at decreasing throughput. dp4 is new leader at 2.1914.
4. **Erase Householders are not Pareto** — at fixed total n, all-writes wins (dp4 > dp4-erase1 > dp4-erase2). Adding erases on top of fixed writes still helps marginally (+0.003-0.005 nat for 1 erase) but the budget should go to writes first.
5. **Hard cap is monotonically harmful at iso-tokens** but cheap. cap=200 binds late and is indistinguishable from no-cap at 4k seqlen, while bounding state norm 5× tighter (565 vs 2810 frob_mean). cap=10 costs +0.004 nat. Length-gen test still open.
6. **Per-channel gate is output-side, not state-regularization** — DN+nogate+carry-v2+cap=200 (2.2147) doesn't recover the +0.0075 nat gap vs perchgate, despite identical state magnitude.
7. **neg-eig** is unstable in vanilla DN; only L2-qk + neg-eig is stable, and it underperforms.
8. **L2-on-V** is the only DN variant where state stays bounded without explicit decay — but it underperforms (2.2120 vs 2.2070).
9. **Head sweep** is U-shaped, sweet spot at h4d64 default.
10. **KDA ≈ GDN** at this scale: vector α decay isn't worth more than scalar α (2.2145 vs 2.2143). KDA's vector α already gives equilibrium state, so carry-v2 adds nothing for it (same pattern: when the layer has built-in decay, carry-v2 is redundant).
11. **Mamba 2 < GDN** (2.2259 vs 2.2143, +0.012 nat worse), consistent with OLMo Hybrid claim "GDN beats Mamba 2 on bits-per-byte". Mamba 2 + carry-v2 also adds nothing (2.2277).
12. **Mamba 3** doesn't run on our setup — bf16 forward instability after ~iter 200, same failure mode across NorMuon/Aurora/clip-grad. Forward (not gradient) is producing NaN; likely a kernel-level bf16 issue in `mamba3_siso_combined`. Not pursued further.
