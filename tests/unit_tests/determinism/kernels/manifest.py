# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Registry of the GPU kernels Megatron-LM dispatches and the determinism tests that cover them.

Every kernel-bearing source file in ``megatron/`` must be registered here, and every
registered kernel must name at least one bit-exact test (or an explicit exemption with a
reason). Two consumers enforce this:

* ``tests/unit_tests/determinism/kernels/test_manifest.py`` -- fails the unit-test bucket
  when a kernel file in the tree is unregistered, a registered path does not exist, or an
  entry has neither tests nor an exemption.
* ``tools/check_kernel_determinism_coverage.py`` -- runs in the ``linting`` CI job on the
  files a PR changes and fails when a kernel-bearing file is unregistered, or when a
  registered kernel source changes without its determinism test changing too (override
  with the ``EXEMPT_LABEL`` PR label).

Keep this module importable with the standard library only: the CI tool loads it by file
path in a job without torch installed.

A *kernel* here is any code path where Megatron itself launches or selects a GPU kernel
whose numerical result could depend on scheduling: Triton ``@jit`` kernels, ``jit_fuser`` /
``torch.compile`` fused functions, C++/CUDA extensions, Transformer Engine and external
library dispatch with algorithm choices (grouped GEMM, causal_conv1d, mamba_ssm, FLA,
DeepEP), and torch ops with a non-deterministic accumulation (``scatter_add_``,
``index_add_``, ``index_put_(accumulate=True)``, ``bincount``, embedding backward).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

# PR label that lets a kernel change merge without touching its determinism test
# (refactors, comment-only edits). Maintainers create it once in the repository.
EXEMPT_LABEL = "determinism-exempt"

# Directories whose every ``.py`` file (``__init__.py`` excluded) must be registered.
KERNEL_DIRECTORIES: Tuple[str, ...] = ("megatron/core/fusions", "megatron/core/ssm/ops")

# Regular expressions (``re.MULTILINE``) that mark a source file as kernel-bearing wherever
# it lives. Matched against the file's current contents by both consumers.
KERNEL_CONTENT_PATTERNS: Tuple[str, ...] = (
    r"@triton\.(jit|autotune|heuristics)",
    r"^\s*(import|from) triton\b",
    r"^\s*(import|from) tilelang\b",
    r"@jit_fuser\b",
    r"\btorch\.compile\b",
    r"\bload_inline\(",
    r"\bCUDAExtension\(",
    r"\.scatter_add_?\(",
    r"\.index_add_?\(",
    r"index_put_?\((?:[^()\n]|\([^()\n]*\))*accumulate\s*=\s*True",
    r"\btorch\.bincount\(",
    r"\btorch\.histc\(",
    r"\bF\.embedding\(",
    # -- external-library kernel dispatch: call sites, never bare imports (a module that
    #    only imports a class or checks isinstance() does not launch anything). Files that
    #    match are registered with ``kind="dispatch"`` (see ``KernelEntry``).
    # Transformer Engine C++ extension (``import transformer_engine_torch as tex``).
    r"\btex\.[A-Za-z_]\w*\(",
    # TE collectives / GEMM / FP8-FP4 master-weight casts / CUDA-graph capture.
    r"\b(gather_along_first_dim|reduce_scatter_along_first_dim)\s*\(",
    r"\b(general_gemm|te_general_gemm|cast_to_fp8|fp8_gemm|cast_master_weights_to_fp8|quantize_master_weights)\s*\(",
    r"\bmake_graphed_callables\s*\(",
    # Fused RoPE (TE / flash-attn / Megatron Triton MLA RoPE) dispatch.
    r"\b(fused_apply_rotary_pos_emb\w*|apply_fused_qkv_rotary_pos_emb|fused_apply_mla_rope_for_\w+|apply_rotary_emb_flash)\s*\(",
    # causal_conv1d / mamba_ssm / flash-linear-attention.
    r"\bcausal_conv1d_(fn|update|update_cuda|varlen_fn|varlen_states|varlen_carry_states|cp)\s*\(",
    r"\b(mamba_chunk_scan_combined\w*|mamba_split_conv1d_scan_combined\w*|selective_state_update|selective_scan_fn)\s*\(",
    r"\b((chunk|fused_recurrent)_gated_delta_(rule|product)\w*|chunk_gdn2|l2_norm|l2norm_(fwd|bwd))\s*\(",
    # DeepEP buffers and cuTile kernels.
    r"\bfused_(dispatch|combine)\s*\(|\b\w*buffer\.(low_latency_)?(dispatch|combine)\s*\(",
    r"@ct\.kernel\b|\bct\.launch\s*\(",
    # FlashInfer (sampling, MXFP8 quantize / GEMM, fused MoE).
    r"\bflashinfer\w*\.[A-Za-z_][\w.]*\(|\bflashinfer_\w+\s*\(|\b(mxfp8_quantize|mm_mxfp8|shuffle_matrix_a|cutlass_fused_moe|get_shuffle_matrix_sf_a_row_indices)\s*\(",
    # apex / TE multi-tensor kernels (scale, l2norm, Adam).
    r"\bmulti_tensor_applier\s*\(",
)


@dataclass(frozen=True)
class KernelEntry:
    """One kernel (or tightly coupled kernel family) and its determinism coverage."""

    name: str
    sources: Tuple[str, ...]
    tests: Tuple[str, ...] = ()
    # Kernel family (``triton``, ``torch.compile``, ``cuda-ext``, ``te-wrapper``, ``torch-op``,
    # ``external-lib``, ``tilelang``) or ``dispatch``: the sources do not define a kernel but
    # select among / call external ones (TE, causal_conv1d, mamba_ssm, FLA, FlashInfer, apex).
    # Dispatch entries are covered by the tests of the kernels they call (named in ``notes``),
    # so ``test_manifest`` does not require their tests to import the source module.
    kind: str = ""
    training_path: bool = True
    notes: str = ""
    # Set (with a reason) only when a bit-exact test is impossible in CI, e.g. the kernel
    # needs hardware or a dependency the CI container lacks. Visible coverage debt.
    exempt_reason: str = ""


# Test-file prefixes used below.
K = "tests/unit_tests/determinism/kernels/"
C = "tests/unit_tests/determinism/correctness/"

KERNELS: Tuple[KernelEntry, ...] = (
    # ---------------------------------------------------------------- fused elementwise (jit_fuser / torch.compile)
    KernelEntry(
        name="fused_bias_swiglu",
        sources=("megatron/core/fusions/fused_bias_swiglu.py",),
        tests=(K + "test_fused_activations.py",),
        kind="torch.compile",
        notes="Elementwise; weighted variants reduce the per-token weight grad over ffn (Inductor tree reduction).",
    ),
    KernelEntry(
        name="fused_bias_geglu",
        sources=("megatron/core/fusions/fused_bias_geglu.py",),
        tests=(K + "test_fused_activations.py",),
        kind="torch.compile",
    ),
    KernelEntry(
        name="fused_bias_gelu",
        sources=("megatron/core/fusions/fused_bias_gelu.py",),
        tests=(K + "test_fused_activations.py",),
        kind="torch.compile",
    ),
    KernelEntry(
        name="fused_weighted_squared_relu",
        sources=("megatron/core/fusions/fused_weighted_squared_relu.py",),
        tests=(K + "test_fused_activations.py",),
        kind="torch.compile",
    ),
    KernelEntry(
        name="fused_bias_dropout_add",
        sources=("megatron/core/fusions/fused_bias_dropout.py",),
        tests=(K + "test_fused_activations.py",),
        kind="torch.compile",
        notes="Consumes the CUDA RNG; replayed under a restored RNG state.",
    ),
    KernelEntry(
        name="compiled_activations",
        sources=(
            "megatron/core/activations.py",
            "megatron/core/transformer/utils.py",
            "megatron/core/transformer/torch_norm.py",
            "megatron/core/transformer/attention.py",
        ),
        tests=(K + "test_fused_activations.py",),
        kind="torch.compile",
        notes="squared_relu/quick_gelu/fast_gelu/tanh_soft_clamp/situ/situ_glu, openai/erf GELU, "
        "L2Norm._norm (row reduction) and Attention._apply_output_gate.",
    ),
    KernelEntry(
        name="fused_vocab_parallel_cross_entropy",
        sources=("megatron/core/fusions/fused_cross_entropy.py",),
        tests=(K + "test_fused_activations.py",),
        kind="torch.compile",
        notes="Rejected by --deterministic-mode (op catalog); the replay test records its status as xfail(strict=False).",
    ),
    KernelEntry(
        name="jit_fuser",
        sources=("megatron/core/jit.py",),
        kind="torch.compile",
        exempt_reason="Decorator only (selects torch.compile / torch.jit.script); every decorated function is registered above.",
    ),
    KernelEntry(
        name="dsv4_q_rms_norm",
        sources=(
            "megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py",
        ),
        tests=(K + "test_fused_activations.py",),
        kind="torch.compile",
        notes="Weightless query RMS norm compiled with torch.compile (row reduction over head_dim); "
        "the rest of the file is orchestration around registered TE / RoPE / CSA kernels.",
    ),
    # ---------------------------------------------------------------- Megatron Triton fusions
    KernelEntry(
        name="fused_pad_routing_map",
        sources=("megatron/core/fusions/fused_pad_routing_map.py",),
        tests=(K + "test_fused_triton_kernels.py",),
        kind="triton",
    ),
    KernelEntry(
        name="fused_indices_to_multihot",
        sources=("megatron/core/fusions/fused_indices_converter.py",),
        tests=(K + "test_fused_triton_kernels.py",),
        kind="triton",
    ),
    KernelEntry(
        name="fused_mla_yarn_rope",
        sources=("megatron/core/fusions/fused_mla_yarn_rope_apply.py",),
        tests=(K + "test_fused_triton_kernels.py",),
        kind="triton",
        notes="Elementwise rotations under timing-based triton.autotune; fwd+bwd replay in sbhd and thd.",
    ),
    KernelEntry(
        name="fused_mhc_kernels",
        sources=(
            "megatron/core/fusions/fused_mhc_kernels.py",
            "megatron/core/transformer/hyper_connection.py",
        ),
        tests=(K + "test_fused_triton_kernels.py",),
        kind="triton",
        notes="Sinkhorn / h_aggregate / h_post_bda / proj_rms_compute_h on the triton, native (torch.compile) and cuTile backends.",
    ),
    # ---------------------------------------------------------------- apex CUDA extensions and local TP layers
    KernelEntry(
        name="fused_layer_norm",
        sources=("megatron/core/fusions/fused_layer_norm.py",),
        tests=(K + "test_tensor_parallel_kernels.py",),
        kind="cuda-ext",
        notes="apex FastLayerNormFN / FusedLayerNormAffineFunction; dgamma/dbeta cross-row reduction.",
    ),
    KernelEntry(
        name="fused_scale_mask_softmax",
        sources=(
            "megatron/core/fusions/fused_softmax.py",
            "megatron/core/transformer/dot_product_attention.py",
        ),
        tests=(K + "test_tensor_parallel_kernels.py",),
        kind="cuda-ext",
        notes="apex scaled_(upper_triang_)masked_softmax_cuda; the local attention path around it is cuBLAS bmm.",
    ),
    KernelEntry(
        name="tensor_parallel_layers",
        sources=("megatron/core/tensor_parallel/layers.py",),
        tests=(K + "test_tensor_parallel_kernels.py",),
        kind="torch-op",
        notes="VocabParallelEmbedding (weight[idx] deterministic branch vs F.embedding) and local Column/RowParallelLinear "
        "incl. apex fused_weight_gradient_mlp_cuda gradient-accumulation fusion.",
    ),
    KernelEntry(
        name="vocab_parallel_cross_entropy",
        sources=("megatron/core/tensor_parallel/cross_entropy.py",),
        tests=(K + "test_tensor_parallel_kernels.py",),
        kind="torch-op",
    ),
    KernelEntry(
        name="tensor_parallel_mappings",
        sources=("megatron/core/tensor_parallel/mappings.py",),
        tests=(C + "test_transformer_layer.py", C + "test_gpt_model.py"),
        kind="external-lib",
        notes="NCCL floating-point reductions pinned by NCCL_ALGO=Ring; covered by the TP/EP/FSDP cells of the model-level suite.",
    ),
    KernelEntry(
        name="tensor_parallel_random_share_storage",
        sources=("megatron/core/tensor_parallel/random.py",),
        kind="cuda-ext",
        exempt_reason="load_inline extension aliases tensor storage on the host; no GPU compute.",
    ),
    # ---------------------------------------------------------------- Transformer Engine wrappers
    KernelEntry(
        name="transformer_engine_wrappers",
        sources=("megatron/core/extensions/transformer_engine.py",),
        tests=(K + "test_te_wrappers.py", C + "test_fp8_determinism.py"),
        kind="te-wrapper",
        notes="TE Linear / LayerNormLinear / Norm / GroupedLinear / DotProductAttention / fused RoPE replayed standalone; "
        "FP8/FP4 recipes in the model-level suite.",
    ),
    KernelEntry(
        name="kitchen_extension",
        sources=("megatron/core/extensions/kitchen.py",),
        kind="te-wrapper",
        exempt_reason="Public stub of the internal Kitchen library; no kernels in this repository.",
    ),
    # ---------------------------------------------------------------- MoE
    KernelEntry(
        name="moe_utils",
        sources=("megatron/core/transformer/moe/moe_utils.py",),
        tests=(K + "test_moe_kernels.py",),
        kind="torch-op",
        notes="permute/unpermute (index_add_ vs scatter_add_), routing (index_put_ vs scatter), sort_chunks, aux loss, "
        "router gating GEMM, TE fused permutation/router kernels.",
    ),
    KernelEntry(
        name="moe_router",
        sources=("megatron/core/transformer/moe/router.py",),
        tests=(K + "test_moe_kernels.py",),
        kind="torch.compile",
    ),
    KernelEntry(
        name="moe_token_dispatchers",
        sources=(
            "megatron/core/transformer/moe/token_dispatcher.py",
            "megatron/core/transformer/moe/moe_layer.py",
        ),
        tests=(K + "test_moe_kernels.py",),
        kind="torch-op",
        notes="MoELayer replay through the allgather / alltoall dispatchers (EP=1, EP=2) and flex+DeepEP when available.",
    ),
    KernelEntry(
        name="moe_experts",
        sources=("megatron/core/transformer/moe/experts.py",),
        tests=(K + "test_moe_kernels.py",),
        kind="te-wrapper",
        notes="TEGroupedMLP / SequentialMLP on uneven expert loads including an empty expert.",
    ),
    KernelEntry(
        name="moe_fused_a2a",
        sources=("megatron/core/transformer/moe/fused_a2a.py",),
        tests=(K + "test_moe_kernels.py",),
        kind="external-lib",
        notes="DeepEP flex dispatcher cell (skipped without deep_ep / >=2 GPUs). HybridEP and NCCL-EP backends are not yet "
        "replayed bit-exactly here.",
    ),
    KernelEntry(
        name="moe_shared_experts",
        sources=("megatron/core/transformer/moe/shared_experts.py",),
        kind="te-wrapper",
        exempt_reason="Composes TE linears and the registered activation fusions; no kernel of its own.",
    ),
    KernelEntry(
        name="moe_paged_stash",
        sources=("megatron/core/transformer/moe/ops/paged_stash.py",),
        tests=("tests/unit_tests/transformer/moe/test_paged_stashing.py",),
        kind="triton",
        notes="Activation page copy kernels (no arithmetic); functional round-trip coverage.",
    ),
    KernelEntry(
        name="moe_batch_invariant_unpermute",
        sources=("megatron/core/transformer/moe/batch_invariant.py",),
        tests=("tests/unit_tests/transformer/moe/test_moe_batch_invariant.py",),
        kind="torch-op",
    ),
    KernelEntry(
        name="inference_routing_mask_kernel",
        sources=("megatron/core/transformer/moe/inference_routing_mask_kernel.py",),
        tests=(K + "test_inference_kernels.py",),
        kind="triton",
        training_path=False,
    ),
    # ---------------------------------------------------------------- SSM
    KernelEntry(
        name="ssm_causal_conv1d",
        sources=("megatron/core/ssm/causal_conv1d.py",),
        tests=(C + "test_ssm_conv1d.py",),
        kind="external-lib",
        notes="Dao-AILab causal_conv1d; channel-last backward reduces dweight/dbias with "
        "atomicAdd unless CAUSAL_CONV1D_DETERMINISTIC (>=1.6.0) picks the workspace path.",
    ),
    KernelEntry(
        name="ssm_mamba2_varlen_kernels",
        sources=(
            "megatron/core/ssm/ops/mamba2/ssd_combined.py",
            "megatron/core/ssm/ops/mamba2/ssd_bmm.py",
            "megatron/core/ssm/ops/mamba2/ssd_chunk_scan.py",
            "megatron/core/ssm/ops/mamba2/ssd_chunk_state.py",
            "megatron/core/ssm/ops/mamba2/ssd_state_passing.py",
            "megatron/core/ssm/ops/mamba2/mamba_ssm.py",
        ),
        tests=(K + "test_ssm_kernels.py",),
        kind="triton",
        notes="mamba_chunk_scan_combined_varlen and selective_state_update under the pinned autotune / ordered workspace.",
    ),
    KernelEntry(
        name="ssm_mamba2_batch_invariant_decode",
        sources=("megatron/core/ssm/ops/mamba2/batch_invariant_decode.py",),
        tests=("tests/unit_tests/ssm/ops/mamba2/test_batch_invariant_decode.py",),
        kind="triton",
        training_path=False,
    ),
    KernelEntry(
        name="ssm_common_kernels",
        sources=(
            "megatron/core/ssm/ops/common/causal_conv1d_triton.py",
            "megatron/core/ssm/ops/common/causal_conv1d_varlen.py",
            "megatron/core/ssm/ops/common/intermediate_extraction.py",
            "megatron/core/ssm/ops/common/determinism.py",
        ),
        tests=(K + "test_ssm_kernels.py",),
        kind="triton",
        notes="determinism.py is the autotune/workspace policy every SSM kernel test runs under.",
    ),
    KernelEntry(
        name="ssm_gdp_kernels",
        sources=(
            "megatron/core/ssm/ops/gdp/chunk.py",
            "megatron/core/ssm/ops/gdp/chunk_h.py",
            "megatron/core/ssm/ops/gdp/chunk_o.py",
            "megatron/core/ssm/ops/gdp/common.py",
            "megatron/core/ssm/ops/gdp/cumsum.py",
            "megatron/core/ssm/ops/gdp/decode_prepare.py",
            "megatron/core/ssm/ops/gdp/fused_recurrent.py",
            "megatron/core/ssm/ops/gdp/metadata.py",
            "megatron/core/ssm/ops/gdp/scaled_dot_kkt.py",
            "megatron/core/ssm/ops/gdp/solve_tril.py",
            "megatron/core/ssm/ops/gdp/wy_fast.py",
            "megatron/core/ssm/context_parallel/gdp.py",
        ),
        tests=(K + "test_ssm_kernels.py",),
        kind="triton",
        notes="Gated Delta Product varlen chunk scan (drives cumsum/l2norm/kkt/solve_tril/wy_fast/chunk_h/chunk_o), "
        "fused recurrent decode and decode-prepare kernels.",
    ),
    KernelEntry(
        name="gated_delta_net",
        sources=(
            "megatron/core/ssm/gated_delta_net/common.py",
            "megatron/core/ssm/gated_delta_net/gdn.py",
            "megatron/core/ssm/gated_delta_net/gdn2.py",
        ),
        tests=(K + "test_ssm_kernels.py", C + "test_hybrid_model.py"),
        kind="torch.compile",
        notes="deterministic_mode selects torch_chunk_gated_delta_rule over FLA (recorded non-deterministic).",
    ),
    KernelEntry(
        name="ssm_triton_cache_manager",
        sources=("megatron/core/ssm/triton_cache_manager.py",),
        kind="triton",
        exempt_reason="Triton compile-cache manager; no kernel.",
    ),
    # ---------------------------------------------------------------- optimizer / distributed
    KernelEntry(
        name="optimizer_kernels",
        sources=(
            "megatron/core/optimizer/clip_grads.py",
            "megatron/core/optimizer/__init__.py",
            "megatron/core/optimizer/optimizer.py",
            "megatron/training/utils/common_utils.py",
        ),
        tests=(K + "test_optimizer_kernels.py",),
        kind="external-lib",
        notes="multi_tensor l2norm / scale (TE, apex or local fallback), including mixed-dtype "
        "caller bucketing and padded-storage replay, and fused Adam; "
        "optimizer.py (gradient unscaling) and training/utils/common_utils.py (param / grad norm "
        "logging) launch the same multi_tensor kernels through multi_tensor_applier.",
    ),
    KernelEntry(
        name="ddp_grad_buffer_reductions",
        sources=("megatron/core/distributed/param_and_grad_buffer.py",),
        tests=(C + "test_gpt_model.py",),
        kind="external-lib",
        notes="NCCL reduce-scatter / all-gather; covered by the FSDP/DP cells of the model-level suite.",
    ),
    KernelEntry(
        name="nccl_allocator",
        sources=("megatron/core/nccl_allocator.py",),
        kind="cuda-ext",
        exempt_reason="Pluggable allocator (ncclMemAlloc); allocation only, no compute kernel.",
    ),
    # ---------------------------------------------------------------- inference (single GPU)
    KernelEntry(
        name="inference_kv_cache_tensor_ops",
        sources=(
            "megatron/core/inference/contexts/attention_context/triton/tensor_ops.py",
            "megatron/core/inference/contexts/fused_kv_append_kernel.py",
        ),
        tests=(K + "test_inference_kernels.py",),
        kind="triton",
        training_path=False,
    ),
    KernelEntry(
        name="inference_moe_kernels",
        sources=(
            "megatron/core/inference/moe/activations.py",
            "megatron/core/inference/moe/batch_invariant.py",
            "megatron/core/inference/moe/permute.py",
            "megatron/core/inference/moe/vllm_fused_moe.py",
            "megatron/core/inference/quantization/mxfp8_quantize.py",
        ),
        tests=(K + "test_inference_kernels.py",),
        kind="triton",
        training_path=False,
        notes="Batch-invariant paths replay bit-exactly; the atomic default unpermute is the negative control. "
        "ordered_reduce_scatter_v (multi-rank) is not replayed here.",
    ),
    KernelEntry(
        name="batch_invariant_kernels",
        sources=("megatron/core/transformer/custom_layers/batch_invariant_kernels.py",),
        tests=(
            K + "test_inference_kernels.py",
            "tests/unit_tests/transformer/test_te_layers_batch_invariant.py",
        ),
        kind="triton",
    ),
    KernelEntry(
        name="mtp_speculative_decoding_kernels",
        sources=("megatron/core/inference/text_generation_controllers/mtp_utils_triton.py",),
        tests=("tests/unit_tests/inference/text_generation_controllers/test_mtp_utils.py",),
        kind="triton",
        training_path=False,
        notes="Integer bookkeeping and state copies checked for exact equality against the torch reference.",
    ),
    KernelEntry(
        name="inference_integer_bookkeeping",
        sources=(
            "megatron/core/inference/contexts/dynamic_context.py",
            "megatron/core/inference/contexts/kv_block_allocator.py",
            "megatron/core/context_parallel/layout.py",
        ),
        kind="torch-op",
        exempt_reason="bincount / scatter_add_ / index_add_ on integer counts: exact regardless of order.",
    ),
    KernelEntry(
        name="inference_nvls_symmetric_memory_collectives",
        sources=(
            "megatron/core/inference/communication/torch_symm_triton/barrier.py",
            "megatron/core/inference/communication/torch_symm_triton/collectives.py",
            "megatron/core/inference/communication/torch_symm_triton/fused_collectives.py",
            "megatron/core/inference/communication/torch_symm_triton/multimem_asm.py",
            "megatron/core/inference/communication/torch_symm_triton/utils.py",
            "megatron/core/inference/communication/torch_symm_triton/variable_collectives.py",
            "megatron/core/inference/moe/metadata.py",
            "megatron/core/inference/symmetric_memory.py",
        ),
        kind="triton",
        training_path=False,
        exempt_reason="Multi-rank NVLS multimem collectives over torch symmetric memory; the in-switch reduction "
        "order is hardware defined and needs an NVLink peer group. Batch-invariant mode routes around them.",
    ),
    KernelEntry(
        name="unified_memory_allocator",
        sources=("megatron/core/inference/unified_memory.py",),
        kind="cuda-ext",
        training_path=False,
        exempt_reason="cudaMallocManaged pluggable allocator; no compute kernel.",
    ),
    KernelEntry(
        name="nvshmem_chunked_copy",
        sources=("megatron/core/resharding/nvshmem_copy_service/kernels/chunked_kernel.cu",),
        kind="cuda-ext",
        training_path=False,
        exempt_reason="Byte copy for refit; needs NVSHMEM and a multi-GPU RL environment.",
    ),
    # ---------------------------------------------------------------- external-library dispatch sites
    # Modules that choose between / call external GPU kernels. Covered by the replay tests of
    # the kernels they call (and by module-level tests where they exist); ``notes`` say which.
    KernelEntry(
        name="ssm_mamba_mixer",
        sources=("megatron/core/ssm/mamba_mixer.py",),
        tests=(C + "test_ssm_conv1d.py", K + "test_ssm_kernels.py"),
        kind="dispatch",
        notes="Selects the pip causal_conv1d / mamba_ssm kernels (causal_conv1d_fn, "
        "mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined, causal_conv1d_update_cuda) or "
        "the Megatron forks (ops/common, ops/mamba2: varlen scan, causal_conv1d_update, "
        "selective_state_update). Module-level replay: test_ssm_conv1d.py::TestMambaMixerDeterminism; "
        "kernels replayed piecewise in test_ssm_kernels.py.",
    ),
    KernelEntry(
        name="ssm_gated_delta_product",
        sources=("megatron/core/ssm/gated_delta_product.py",),
        tests=(K + "test_ssm_kernels.py",),
        kind="dispatch",
        notes="Dispatches FLA chunk_gated_delta_product / l2_norm, the CuTeDSL gdp_attn kernel "
        "(gdp_cutedsl_kernel; not in the CI container, uncovered), causal_conv1d and the Megatron "
        "GDP forks (chunk_gated_delta_product_varlen, fused_recurrent_gated_delta_rule_update), all "
        "replayed in test_ssm_kernels.py. No module-level replay yet (HYBRID_CONFIGS has no GDP cell).",
    ),
    KernelEntry(
        name="rope_dispatch",
        sources=("megatron/core/models/common/embeddings/rope_utils.py",),
        tests=(K + "test_te_wrappers.py",),
        kind="dispatch",
        notes="apply_rotary_pos_emb selects TE fused RoPE (replayed in test_te_wrappers.py), "
        "flash-attn apply_rotary_emb (not in the CI container) or the unfused torch path.",
    ),
    KernelEntry(
        name="mla_rope_dispatch",
        sources=(
            "megatron/core/transformer/multi_latent_attention.py",
            "megatron/core/transformer/experimental_attention_variant/absorbed_mla.py",
        ),
        tests=(K + "test_fused_triton_kernels.py", K + "test_te_wrappers.py"),
        kind="dispatch",
        notes="Calls the Triton MLA YaRN RoPE kernels (fused_apply_mla_rope_for_q / _kv, replayed in "
        "test_fused_triton_kernels.py) and TE fused RoPE (test_te_wrappers.py). The TE "
        "FusedMLAQUpProj GEMM path and mxfp8_quantize_only are not replayed.",
    ),
    KernelEntry(
        name="fp8_fp4_master_weight_casts",
        sources=(
            "megatron/core/fp8_utils.py",
            "megatron/core/fp4_utils.py",
            "megatron/core/distributed/fsdp/src/megatron_fsdp/mixed_precision.py",
        ),
        tests=(C + "test_fp8_determinism.py", K + "test_optimizer_kernels.py"),
        kind="dispatch",
        notes="TE cast_master_weights_to_fp8 / cast_to_fp8 (covered at model level by "
        "test_fp8_determinism.py), NVFP4 quantize_master_weights (needs TE>=2.7 + Blackwell, "
        "uncovered) and the apex multi_tensor_scale used by Megatron-FSDP mixed precision "
        "(kernel replayed in test_optimizer_kernels.py).",
    ),
    KernelEntry(
        name="inference_mxfp8_quantization",
        sources=(
            "megatron/core/inference/quantization/mxfp8_tensor.py",
            "megatron/core/inference/quantization/utils.py",
            "megatron/core/inference/moe/flashinfer_mxfp8.py",
        ),
        tests=(K + "test_inference_kernels.py",),
        kind="dispatch",
        training_path=False,
        notes="MXFP8Tensor dispatches the Megatron quantize kernel (replayed: "
        "test_mxfp8_quantize_replays) or FlashInfer mxfp8_quantize / mm_mxfp8 / fused MoE, which need "
        "flashinfer on Blackwell and are not replayed in CI.",
    ),
    KernelEntry(
        name="inference_flashinfer_sampling",
        sources=("megatron/core/inference/sampling/flashinfer_sampling.py",),
        kind="dispatch",
        training_path=False,
        exempt_reason="FlashInfer top-k / top-p sampling kernels driven by an explicit torch.Generator "
        "with deterministic=True; a seed-restore replay test needs flashinfer in the unit-test "
        "container and is tracked as a follow-up.",
    ),
    KernelEntry(
        name="inference_tp_layers",
        sources=("megatron/core/tensor_parallel/inference_layers.py",),
        kind="dispatch",
        training_path=False,
        exempt_reason="Inference-only TP layers composing tex.rmsnorm_fwd (the TE norm kernel is "
        "replayed in test_te_wrappers.py), FlashInfer MXFP8 GEMM (Blackwell only) and TE "
        "gather / reduce-scatter collectives.",
    ),
    KernelEntry(
        name="te_tp_collectives",
        sources=("megatron/core/tensor_parallel/generalized_tensor_parallelism.py",),
        kind="dispatch",
        exempt_reason="TE gather_along_first_dim / reduce_scatter_along_first_dim of weights and "
        "wgrads in generalized tensor parallelism; multi-rank NCCL collectives pinned by "
        "NCCL_ALGO=Ring under --deterministic-mode, not exercised by the determinism model tests.",
    ),
    KernelEntry(
        name="te_thd_partitioned_indices",
        sources=(
            "megatron/core/datasets/data_schedule.py",
            "megatron/core/ssm/mamba_context_parallel.py",
            "megatron/core/utils.py",
            "megatron/core/models/mimo/partition/utils.py",
            "megatron/core/models/multimodal/llava_model.py",
            "megatron/rl/rl_utils.py",
        ),
        kind="dispatch",
        exempt_reason="tex.thd_get_partitioned_indices computes integer sequence partitions for THD "
        "context parallelism; exact regardless of execution order.",
    ),
    KernelEntry(
        name="cuda_graph_capture",
        sources=("megatron/core/transformer/cuda_graphs.py",),
        kind="dispatch",
        exempt_reason="TE make_graphed_callables captures and replays kernels that are registered on "
        "their own; the capture order is fixed by the callable list and adds no numerics.",
    ),
    # ---------------------------------------------------------------- DeepSeek sparse attention (TileLang)
    KernelEntry(
        name="dsa_tilelang_kernels",
        sources=(
            "megatron/core/transformer/experimental_attention_variant/ops/indexer.py",
            "megatron/core/transformer/experimental_attention_variant/ops/sparse_mla.py",
            "megatron/core/transformer/experimental_attention_variant/ops/tilelang_dsa.py",
            "megatron/core/transformer/experimental_attention_variant/ops/tilelang_indexer_bwd.py",
            "megatron/core/transformer/experimental_attention_variant/ops/tilelang_indexer_fwd.py",
            "megatron/core/transformer/experimental_attention_variant/ops/tilelang_indexer_loss.py",
            "megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_bwd.py",
            "megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_fwd.py",
            "megatron/core/transformer/experimental_attention_variant/ops/tilelang_utils.py",
            "megatron/core/transformer/experimental_attention_variant/dsa_kernels.py",
            "megatron/core/transformer/experimental_attention_variant/dsa_tilelang_kernels.py",
        ),
        kind="tilelang",
        exempt_reason="Experimental DeepSeek sparse attention kernels (indexer / sparse MLA backward accumulate with "
        "atomics). Bit-exact replay is tracked as a follow-up; parity tests live in "
        "tests/unit_tests/transformer/experimental_attention_variant/.",
    ),
)


def entries_for(path: str) -> Tuple[KernelEntry, ...]:
    """Return the entries that list ``path`` among their sources."""
    return tuple(entry for entry in KERNELS if path in entry.sources)
