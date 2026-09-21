# Qwen3.8-Flash-Next (`qwen4_exp`) in Megatron Core

Training support for the Qwen3.8-Flash-Next text model (HF `model_type: qwen4_exp`, the Qwen4
architecture preview): 48 hybrid layers of Gated DeltaNet (3 per block) and Qwen Sparse Attention
(1 per block), a 512-expert MoE with a gated shared expert on every layer, four gated-residual
streams, a hashed n-gram memory (PLE) on decoder layer 2 with a ~51 B-parameter table, and one
multi-token-prediction depth. The vision tower of the released checkpoint is not modelled.

This folder is the entry point for the `lit/main_qwen4` branch. It ties together the three component
documents that live one level up and adds what is specific to assembling and running the whole
model.

> **Want to run something rather than read?** Go to
> [`examples/qwen3.8_flash_next/`](../../../examples/qwen3.8_flash_next/). It holds the runnable
> counterpart of everything here: a self-contained parity harness that checks this implementation
> against Hugging Face `qwen4_exp` on one GPU in about five minutes, and launch scripts for the
> proxy and the full model. These documents record what was measured; that directory is how you
> reproduce it.

| Document | Contents |
|---|---|
| [`architecture/config_mapping.md`](architecture/config_mapping.md) | HF `config.json` → Megatron arguments, field by field; the hybrid layer pattern; the settings that are easy to get wrong (router path, PLE width, GDN gate, MTP norms) |
| [`training/proxy_single_node.md`](training/proxy_single_node.md) | An 8-layer / 32-expert proxy with every per-layer dimension real; one `torchrun` on a single node; what to expect |
| [`training/full_model_64gpu.md`](training/full_model_64gpu.md) | The 48-layer model on 64 GPUs (EP64, BF16): the full argument list, measured memory and step time, from-scratch schedule |
| [`checkpoint/hf_weights_and_resume.md`](checkpoint/hf_weights_and_resume.md) | Weight-mapping rules from the released checkpoint, the native `torch_dist` checkpoint, and the resume flags; what was measured |
| [`validation/support_matrix.md`](validation/support_matrix.md) | **What the composed model actually supports** — EP/TP/PP/VPP/CP/THD/recompute/MXFP8/CUDA-graph/FSDP, each measured on the proxy, with the TODO list the gaps produce |
| [`validation/parity_and_acceptance.md`](validation/parity_and_acceptance.md) | Numerical parity against the HF implementation (random-init proxy and real weights), the acceptance matrix, and what is still unverified |
| [`integration/branches_and_rebuild.md`](integration/branches_and_rebuild.md) | How the branch is built from three feature branches, what the one remaining glue commit is, and how to rebuild it |

Component documents (same directory level): [`../gated_residual.md`](../gated_residual.md),
[`../qsa.md`](../qsa.md), [`../engram.md`](../engram.md). Gated DeltaNet, hybrid stacks, MoE and
MTP are stock Megatron Core.

## Status (2026-09-15)

| Area | State |
|---|---|
| Model definition | complete; every HF `config.json` field of the text model has a Megatron argument (`architecture/`) |
| Numerical parity vs HF | whole-model proxy: logits `max_abs 2.7e-7`, 161/161 parameter gradients, 20-step trajectory; real weights (4-layer truncation): every point before the first MoE router at fp32 noise, routing ties after it (`validation/`) |
| Released weights | convert strictly (all 131 shards consumed), load exactly as a native `torch_dist` checkpoint at EP64, 50-step resume on 64 GB300s (`checkpoint/`) |
| Memory / layout | BF16, expert parallelism only: EP64 peaks at 204 GiB per GPU with 29 GiB headroom; EP32 does not fit without recompute (`training/full_model_64gpu.md`) |
| Parallelism / features | EP, TP+SP, PP, VPP, CP (unpacked), packed rows (`--sft`), recompute and MXFP8 all **measured working** on the composed stack; CUDA graphs, FSDP and packed-rows+CP are rejected at startup ([`validation/support_matrix.md`](validation/support_matrix.md)) |
| Not yet done | loss quality on real text (all runs used mock data); performance tuning (hybridep, CUDA graphs, fused GR kernels, QSA sparse kernel by default); CP on packed rows; the multimodal variant |

## Entry point and specs

The model is built by `pretrain_hybrid.py` with `--hybrid-layer-pattern` and the config-aware
spec `gated_residual_hybrid_stack_spec` (`megatron/core/models/hybrid/hybrid_layer_specs.py`),
which composes the QSA-extended hybrid stack with the norm-free gated-residual layers. The n-gram
memory is attached by `HybridModelBuilder` from `--engram-*` arguments. The resulting
`HybridModel` accepts `input_ids` on every pipeline stage (the memory hashes raw tokens), and the
hyper-connection wrapper adds the memory's output to the residual streams before its read gate.
