# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
from dataclasses import dataclass, field
from typing import Literal

SubmissionGranularity = Literal["R", "G", "B"]
ConsumptionGranularity = Literal["G", "B"]

_BOOL_OPTIONAL = {"argparse_meta": {"action": argparse.BooleanOptionalAction}}


@dataclass(kw_only=True)
class RLConfig:
    """Configuration settings for the built-in RL training loop (GRPO)."""

    perform_rl_step: bool = False
    """Use the RL training step."""

    rl_prompts_per_eval: int = 32
    """Number of prompts to evaluate for each RL task. This evaluation can be very expensive when
    using environments that evaluate pass@k so we default to a lower number."""
    # TODO(rkirby): allow for "complete" evaluation when rl_prompts_per_eval is set to -1

    grpo_prompts_per_step: int = 32
    """Number of GRPO groups (G in the paper)."""

    grpo_group_size: int = 2
    """Number of samples per GRPO group."""

    rl_generation_lag: float | None = None
    """Number of trainer batches of rollout generation lag to allow. The number of in-flight trainer
    batches is this value plus one. May be fractional or negative; the minimum of -1 keeps a single
    unit of generation work in flight. If omitted, the lag is autotuned to the inference engine's
    request capacity when rl_partial_rollouts is set, and is 0 otherwise. Requires
    rl_partial_rollouts when greater than 0. Mutually exclusive with rl_max_inflight_requests; both
    are accepted only when they agree, which is the case when a normalized config is reloaded."""

    rl_max_inflight_requests: int | None = None
    """Maximum number of inference requests RL generation may keep in flight: equivalent to
    (rl_generation_lag + 1) training batches of grpo_prompts_per_step * grpo_group_size requests
    each. Requires rl_partial_rollouts when above one training batch; mutually exclusive with
    rl_generation_lag."""

    rl_submission_granularity: SubmissionGranularity = "B"
    """Granularity for submitting rollout generation work. R submits individual rollouts
    independently while still yielding complete rollout groups to training. G submits one rollout
    group at a time. B submits grpo_prompts_per_step rollout groups together."""

    rl_consumption_granularity: ConsumptionGranularity = "B"
    """Granularity for consuming generated rollout groups. G consumes groups as they complete.
    B consumes complete trainer batches in submission order. R is not supported."""

    rl_durable_rollout_bank: bool = False
    """Persist completed rollout groups to a durable, write-through ledger so they survive a SIGKILL
    (the SLURM time limit) and are restored at restart instead of regenerated. No-op when unset."""

    rl_rollout_bank_dir: str | None = None
    """Directory for the durable rollout bank (on Lustre). Defaults to <save>/rollout_bank so the
    bank stays coupled to the checkpoint."""

    rl_rollout_bank_max_bytes: int = 0
    """Soft cap (bytes) on the rollout bank size; 0 = unbounded. On exceed, a warning is logged.
    Compaction occurs at the next checkpoint regardless of the cap and never blocks generation."""

    grpo_iterations: int = 2
    """Number of iterations per GRPO implementation."""

    grpo_clamp_eps_lower: float = 0.01
    """Lower GRPO clipping bound. As in DAPO the lower and upper bounds may differ; set them equal
    for vanilla GRPO."""

    grpo_clamp_eps_upper: float = 0.01
    """Upper GRPO clipping bound. In the vanilla implementation, equals the lower one."""

    grpo_kl_beta: float = 0.001
    """KL term weight in the GRPO loss."""

    grpo_entropy_term_weight: float = 0.0
    """Entropy term weight in the GRPO loss."""

    grpo_filter_groups_with_same_reward: bool = False
    """Filter groups with same reward."""

    langrl_env_config: str | None = None
    """Path to YAML config file for RL environment configuration."""

    rl_default_temperature: float = 1.0
    """Default temperature for model inference."""

    rl_default_top_p: float = 0.0
    """Default top-p for model inference."""

    rl_default_top_k: int = -1
    """Default top-k for model inference."""

    rl_offload_optimizer_during_inference: bool = False
    """Offload optimizer state to CPU during inference/rollout to save GPU memory."""

    rl_kv_cache_management_mode: Literal["persist", "offload", "recompute"] = "persist"
    """KV cache management mode during RL training. persist: leave KV cache in GPU memory (default).
    offload: offload KV cache to CPU during training. recompute: deallocate KV cache and recompute
    from scratch each cycle."""

    rl_persist_cuda_graphs: bool = field(default=False, metadata=_BOOL_OPTIONAL)
    """Persist CUDA graphs when the inference engine is suspended. If False, CUDA graphs are deleted
    on suspend and re-captured on resume."""

    rl_partial_rollouts: bool = field(default=False, metadata=_BOOL_OPTIONAL)
    """Allow inference to continue generating rollouts while training updates the policy weights.
    This enables off-policy training where rollouts may be generated with a stale version of the
    policy. Use rl_generation_lag to control the degree of staleness."""

    rl_inference_logprobs_is_correction: bool = field(default=False, metadata=_BOOL_OPTIONAL)
    """If set, use inference logprobs in importance sampling correction of the loss."""

    rl_importance_sampling_truncation_coef: float | None = None
    """If rl_inference_logprobs_is_correction is on and this coefficient is set, apply truncation
    for the IS correction at GRPO loss."""

    rl_use_sequence_packing: bool = field(default=False, metadata=_BOOL_OPTIONAL)
    """Enable sequence packing."""

    rl_sequence_packing_max_sequences_per_bin: int = 50
    """Maximum number of sequences that can be packed into a single bin."""

    rl_sequence_packing_algo: Literal["fifo", "round-robin"] = "fifo"
    """Algorithm for distributing packed bins across ranks. fifo: first-in-first-out sequential
    distribution. round-robin: distribute bins cyclically across ranks for better load balancing."""

    rl_training_cuda_graphs: bool = field(default=False, metadata=_BOOL_OPTIONAL)
    """If set, do not toggle CUDA graphs on/off between inference and training phases."""

    rl_inference_tensor_model_parallel_size: int | None = None
    """Degree of tensor model parallelism for inference for RL."""

    rl_inference_pipeline_model_parallel_size: int | None = None
    """Degree of pipeline model parallelism for inference for RL."""

    rl_inference_expert_model_parallel_size: int | None = None
    """Degree of expert model parallelism for inference for RL."""

    rl_inference_expert_tensor_model_parallel_size: int | None = None
    """Degree of expert tensor model parallelism for inference for RL. For MoE models, this controls
    the TP size for expert layers specifically. Defaults to training expert_tensor_parallel_size if
    not specified."""

    rl_inference_model_unified_memory_level: Literal[0, 1] = 0
    """Allocate the separate RL inference model parameters from a unified virtual memory (UVM) CUDA
    mempool. Level 0 disables UVM (default). Level 1 enables UVM allocation so the inference model
    weights can be prefetched to CPU when idle while keeping CUDA-graph-safe device pointers."""

    rl_offload_inference_model_weights: bool = field(default=False, metadata=_BOOL_OPTIONAL)
    """When using a separate RL inference model, offload its weights to CPU when not doing rollout
    inference, and restore to GPU right before inference. Works with two backends: 1) UVM (when
    rl_inference_model_unified_memory_level=1), or 2) torch_memory_saver (when UVM is not enabled;
    requires torch_memory_saver to be installed)."""

    refit_method: Literal["nccl", "nccl_m2n", "gloo", "nvshmem", "nixl"] = "gloo"
    """Method to refit model weights. nccl: use NCCLCopyService; nccl_m2n: use the official NCCL M2N
    API from a non-RL launcher such as the ReFIT benchmark; gloo: use GlooCopyService over CPU;
    nvshmem: use NVSHMEMCopyService; nixl: use NixlCopyService."""

    refit_execution_batch_bytes: int | None = None
    """Optional soft per-rank byte limit for ReFIT execution staging. The default None preserves one
    model-wide generic submission and NCCL M2N's existing 256 MiB default."""

    rl_verify_model_weights_swap: bool = field(default=False, metadata=_BOOL_OPTIONAL)
    """If set, verify that the model weights were correctly transferred by comparing forward pass
    outputs on the first swap of model weights."""

    rl_skip_bos_token: bool = field(default=False, metadata=_BOOL_OPTIONAL)
    """Skip BOS token at the beginning of the sequences."""

    rl_profile: bool = False
    """Enable RL profiling to collect detailed timer data (JSONL + CSV)."""

    rl_profile_dir: str | None = None
    """Directory to write RL profiling data. Defaults to {save}/profiles."""

    rl_inference_parsers: list[str] = field(
        default_factory=list, metadata={"argparse_meta": {"nargs": "*"}}
    )
    """List of response parsers to enable for RL inference (e.g. deepseek-r1-reasoning
    qwen3-coder-tool)."""

    grpo_samples_per_iteration: int = field(init=False)
    """Derived: grpo_prompts_per_step * grpo_group_size, the samples generated per RL iteration."""

    def __post_init__(self):
        self.grpo_samples_per_iteration = self.grpo_prompts_per_step * self.grpo_group_size

        if self.refit_execution_batch_bytes is not None:
            assert self.refit_execution_batch_bytes > 0, (
                "--refit-execution-batch-bytes must be a positive integer"
            )

        # The remaining checks describe the RL loop and only apply when it runs.
        if not self.perform_rl_step:
            return

        assert self.refit_method != "nccl_m2n", "nccl_m2n is unsupported by the built-in RL loop"

        if self.rl_max_inflight_requests is not None:
            assert self.rl_max_inflight_requests >= 1, (
                f"--rl-max-inflight-requests ({self.rl_max_inflight_requests}) must be >= 1."
            )
            if self.rl_max_inflight_requests > self.grpo_samples_per_iteration:
                assert self.rl_partial_rollouts, (
                    "--rl-max-inflight-requests above one training batch "
                    f"({self.grpo_samples_per_iteration} requests) requires --rl-partial-rollouts."
                )
            # Total in-flight requests = (lag + 1) trainer batches of P * G requests each.
            derived_lag = self.rl_max_inflight_requests / self.grpo_samples_per_iteration - 1
            assert self.rl_generation_lag is None or self.rl_generation_lag == derived_lag, (
                "--rl-generation-lag and --rl-max-inflight-requests are mutually exclusive."
            )
            self.rl_generation_lag = derived_lag
        if self.rl_generation_lag is None:
            # With rl_partial_rollouts the lag is autotuned from engine capacity at inference
            # launch; otherwise generation is fully synchronous.
            if not self.rl_partial_rollouts:
                self.rl_generation_lag = 0.0
        else:
            assert self.rl_generation_lag >= -1, (
                f"--rl-generation-lag ({self.rl_generation_lag}) must be >= -1."
            )
            if self.rl_generation_lag > 0:
                assert self.rl_partial_rollouts, (
                    "--rl-generation-lag requires --rl-partial-rollouts."
                )

        assert self.rl_submission_granularity == "B" or self.rl_partial_rollouts, (
            f"--rl-submission-granularity {self.rl_submission_granularity} requires "
            "--rl-partial-rollouts."
        )
        assert self.rl_consumption_granularity != "R", (
            "--rl-consumption-granularity R is not currently supported."
        )
        assert not (
            self.rl_submission_granularity == "B" and self.rl_consumption_granularity == "G"
        ), "--rl-submission-granularity B with --rl-consumption-granularity G is not supported."

        # KV cache offload requires CUDA graph persistence: recapturing CUDA graphs runs dummy
        # forward passes that corrupt the preserved KV data.
        assert self.rl_kv_cache_management_mode != "offload" or self.rl_persist_cuda_graphs, (
            "--rl-kv-cache-management-mode=offload requires --rl-persist-cuda-graphs"
        )

        if (
            self.rl_offload_inference_model_weights
            and self.rl_inference_model_unified_memory_level != 1
        ):
            # Not using UVM, so torch_memory_saver must provide the CPU backup.
            try:
                import torch_memory_saver  # noqa: F401
            except ImportError:
                raise AssertionError(
                    "To use --rl-offload-inference-model-weights without UVM "
                    "(--rl-inference-model-unified-memory-level=1), `torch_memory_saver` must be "
                    "installed. See https://github.com/fzyzcjy/torch_memory_saver."
                )
