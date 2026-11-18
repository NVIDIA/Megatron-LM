# V4.1 integration snapshot

This branch assembles the V4.1 text training components for evaluation. Start with
[the training guide](examples/deepseek_v41/README.md). The training entrypoint is
`pretrain_hybrid.py`; the model uses `--dsv4-version v4.1` and physical `VE` layer
pairs. Model weights and tokenizer-derived Engram maps are not included.

## Included components

| Component | Upstream PR |
| --- | --- |
| Generic cross-layer tensor state, checkpointing, graphs and PP/VPP transport | [#7678](https://github.com/NVIDIA/Megatron-LM/pull/7678) |
| Single-pass mHC and parameter-publication readiness | [#8223](https://github.com/NVIDIA/Megatron-LM/pull/8223) |
| CSA2 Full/Reindex/Reuse, candidate indexing and MXFP8 selection | [#8224](https://github.com/NVIDIA/Megatron-LM/pull/8224) |
| V4.1 Engram on shared EP embedding tables | [#8225](https://github.com/NVIDIA/Megatron-LM/pull/8225) |
| V4.1 model composition and FP8 recompute/capture integration | [#8226](https://github.com/NVIDIA/Megatron-LM/pull/8226) |
| Training builder and static packed pipeline scheduling | [#8227](https://github.com/NVIDIA/Megatron-LM/pull/8227) |
| Optional Mixture-of-Kittens MoE backend | [#8210](https://github.com/NVIDIA/Megatron-LM/pull/8210) |

The MoK dependency is pinned to the code from PR #8210 at
`077837249a1f128ca02cf58f9fca2ebc180e8fb8`, plus the small configuration guard that
keeps native HybridEP padding requirements out of MoK dispatch. Newer revisions
of that PR are not part of this validated snapshot. The MoK guard remains an
integration follow-up owned by the MoK effort.

This is an integration snapshot, not a new pull request or a released-model
convergence claim. Component PRs retain their separate review boundaries. The
production and test files are identical to the validated integrated commit
`4fe0301dfccf5a3d689d2efc7cf5476a81be475f`; publication only reorganizes commit
ancestry and updates documentation.

## Validation scope

The integrated stack completed BF16 and MXFP8 training on 16 GB200 GPUs with
TP1/PP2/VPP4/EP8/CP4, 24 logical blocks, 48 experts, 16K packed documents, MBS1 and
GBS128. The recipe uses distributed Muon, MoK, selective MLA-up-projection/mHC
recomputation and overlapped gradient reduction/parameter gather. Engram and
CUDA graphs are disabled in this performance recipe.

Each run completed 12 iterations with no skipped or non-finite logged steps.
Using the median of iterations 3–12 and the same model-FLOPs convention:

| Precision | Seconds/step | Model TFLOPS/GPU | Peak PyTorch allocated memory |
| --- | ---: | ---: | ---: |
| BF16 | 27.28 | 308.5 | 128.2 GiB |
| MXFP8 | 29.09 | 289.3 | 117.4 GiB |

These are proxy measurements in the recorded GB200 environment, not full-model
performance or a hardware-counter measurement. PyTorch allocated memory excludes
external CUDA allocations. The tested software stack used PyTorch nv26.09,
Transformer Engine 2.20.2 and cuDNN Frontend 1.29.

Separate small-configuration checks cover MXFP8 selection, packed/CP boundaries,
explicit graph replay, recomputation, PP and Engram/EP. They do not establish
full-model, same-weight cross-branch gradient equivalence. Supported execution
boundaries and remaining exclusions are described in the training guide.
