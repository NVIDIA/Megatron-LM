#!/usr/bin/env bash
# DeepSeek-V4.1-Flash DAPO launcher: mLite deployment-operator actor (impl=vllm) and a
# resident vLLM BI=1 rollout, synced online every step in the release checkpoint format.
#
# Train/inference consistency contract (BI=1):
#   * actor forward calls the rollout's own ops (megatron.lite.model.deepseek_v41.vllm);
#   * resync = deepseek_v41.lite.export (block-FP8 / MXFP4 from the same quantizers the actor
#     forward applies) -> vLLM layerwise checkpoint reload (verl_mlite.compat), so after each
#     sync the rollout holds the actor's per-step fake-quant weights byte for byte;
#   * frozen Engram tables are not synced: the rollout loads them from the release
#     (load_format=auto), the actor reads the same release rows.
set -euo pipefail

# ---------------------------------------------------------------------------
# Inputs and geometry
# ---------------------------------------------------------------------------
: "${MODEL_PATH:?set MODEL_PATH to the DeepSeek-V4.1-Flash release checkpoint}"
: "${TRAIN_FILES:?set TRAIN_FILES (DAPO-Math-17K parquet)}"
: "${VAL_FILES:?set VAL_FILES (AIME 2024 parquet)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

NNODES="${NNODES:-16}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
TOTAL_GPUS=$((NNODES * NGPUS_PER_NODE))
ACTOR_PP="${ACTOR_PP:-1}"
ACTOR_CP="${ACTOR_CP:-1}"
ACTOR_EP="${ACTOR_EP:-${TOTAL_GPUS}}"
# One rollout replica per node: DP=EP=4 over the node's GPUs (DEP4).
ROLLOUT_DP="${ROLLOUT_DP:-${NGPUS_PER_NODE}}"
ROLLOUT_EP="${ROLLOUT_EP:-${NGPUS_PER_NODE}}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_SEQ_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-4096}"
# Each Ray worker imports vLLM via the verl_mlite sitecustomize (~50 s); cap the prestarted pool.
RAY_NUM_CPUS="${RAY_NUM_CPUS:-null}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
# DSpark speculative decoding (drafter frozen at the release; target-only sync).
ROLLOUT_SPEC="${ROLLOUT_SPEC:-True}"
# Level 2 uses native vLLM preservation for frozen drafter/router state; requires
# vLLM 00474aeec6 or newer (reload permutations stay on host).
ROLLOUT_SLEEP_LEVEL="${ROLLOUT_SLEEP_LEVEL:-2}"
# BATCH_INVARIANT=0 changes only the rollout: VLLM_BATCH_INVARIANT=0 and vLLM default kernels.
BATCH_INVARIANT="${BATCH_INVARIANT:-1}"
ROLLOUT_FULL_DETERMINISM="${ROLLOUT_FULL_DETERMINISM:-$([[ $BATCH_INVARIANT == 1 ]] && echo True || echo False)}"
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-5120}"
LOG_PROB_MAX_TOKEN_LEN="${LOG_PROB_MAX_TOKEN_LEN:-${MAX_SEQ_LEN}}"
export DSV41_ATTENTION_CORE="${DSV41_ATTENTION_CORE:-flashmla}"
export DSV41_HEAD_DTYPE="${DSV41_HEAD_DTYPE:-float32}"
INDEXER_KV_DTYPE="${INDEXER_KV_DTYPE:-mxfp4}"
INDEXER_SPARSE_LOGITS="${INDEXER_SPARSE_LOGITS:-false}"
# chat: "<User>..<Assistant></think>" (no reasoning); thinking: effort-75 reasoning prefix.
PROMPT_MODE="${PROMPT_MODE:-chat}"

# Parameters stay resident; optimizer states offload in chunks. Independent gradient
# release is opt-in until its separate RL gate passes.
OPTIMIZER_STATE_CHUNK_MB="${OPTIMIZER_STATE_CHUNK_MB:-2048}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
GRAD_OFFLOAD="${GRAD_OFFLOAD:-False}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-5}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
# AIME 2024: 8 samples per problem at the rollout temperature, 4096-token output budget
# (fixed; does not follow MAX_RESPONSE_LENGTH).
VAL_N="${VAL_N:-8}"
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-4096}"
RESUME_MODE="${RESUME_MODE:-auto}"

[[ "${VERBOSE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
EXAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -L)"
LITE_ROOT="$(cd "${EXAMPLE_ROOT}/../.." && pwd -L)"
REPO_ROOT="$(cd "${LITE_ROOT}/../.." && pwd -L)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXAMPLE_ROOT}/outputs/ds41_dapo}"
export PYTHONPATH="${EXAMPLE_ROOT}:${LITE_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Batch-invariant execution (sparse FlashMLA rollout; the actor reads the same switch)
# ---------------------------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1 VLLM_USE_V1=1
# Transformers has no deepseek_v41 config class; register an opaque one in every process.
export VERL_MLITE_HF_CONFIG_MODEL_TYPE=deepseek_v41
export VERL_MLITE_ROLLOUT_SLEEP_LEVEL="${ROLLOUT_SLEEP_LEVEL}"
export VERL_MLITE_VAL_MAX_TOKENS="${VAL_MAX_TOKENS}"
export VLLM_BATCH_INVARIANT=1 VERL_FULL_DETERMINISM=1
export VLLM_BI_MXFP8_GEMM=tensor_core VLLM_BI_BF16_GEMM=tensor_core
export VLLM_BI_MXFP8_PREQUANT=1 VLLM_BI_MHC_OVERLAP=1
case "$DSV41_ATTENTION_CORE" in
  flashmla) ATTENTION_BACKEND=FLASHMLA_SPARSE_DSV41; export VLLM_BI_MEGA_ATTN=0 ;;
  mega) ATTENTION_BACKEND=FLASHMLA_MEGA_ATTN_DSV41; export VLLM_BI_MEGA_ATTN=1 ;;
  *) echo "DSV41_ATTENTION_CORE must be flashmla or mega" >&2; exit 2 ;;
esac
case "$DSV41_HEAD_DTYPE" in
  float32|bfloat16) ;;
  *) echo "DSV41_HEAD_DTYPE must be float32 or bfloat16" >&2; exit 2 ;;
esac
[[ "$ROLLOUT_SLEEP_LEVEL" =~ ^[12]$ ]] || { echo "ROLLOUT_SLEEP_LEVEL must be 1 or 2" >&2; exit 2; }
export VLLM_USE_V2_MODEL_RUNNER=1
if [[ "$BATCH_INVARIANT" == 0 ]]; then export VERL_MLITE_ROLLOUT_ENV=VLLM_BATCH_INVARIANT=0; fi
export VLLM_BI_MEGA_GATE="${VLLM_BI_MEGA_GATE:-0}" VLLM_BI_MEGA_MHC="${VLLM_BI_MEGA_MHC:-0}"
export MCORE_DIST_CKPT_STREAM_D2H=0  # Use the unmodified Core checkpoint writer.
export VLLM_BI_REPLAY_STAGE_LAYERS=2
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-0}" NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export VERL_MLITE_SKIP_RUNTIME_PATCHES=0

case "${PROMPT_MODE}" in
  chat|thinking) ;;
  *) echo "PROMPT_MODE must be chat or thinking" >&2; exit 2 ;;
esac
# Single-turn renderings of the release encoding/encoding.py (encode_messages).
CHAT_TEMPLATE="$(<"${SCRIPT_DIR}/deepseek_v41_${PROMPT_MODE}_template.jinja")"

case "${ROLLOUT_SPEC,,}" in
  true|1) SPEC_ARGS=("+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config={method:dspark,num_speculative_tokens:5,draft_sample_method:probabilistic,rejection_sample_method:block,enable_adaptive_verification:false}") ;;
  false|0) SPEC_ARGS=() ;;
  *) echo 'ROLLOUT_SPEC must be true or false' >&2; exit 2 ;;
esac
SLEEP_PRESERVE="'*.bias_vl'"
[[ ${#SPEC_ARGS[@]} -gt 0 ]] && SLEEP_PRESERVE+=",'draft.*'"
SPEC_ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.sleep_preserve_parameter_names=[${SLEEP_PRESERVE}]")

RUN_NAME="${RUN_NAME:-ds41_dapo_${TOTAL_GPUS}gpu_ep${ACTOR_EP}_cp${ACTOR_CP}_dep${ROLLOUT_DP}_bs${TRAIN_BATCH_SIZE}x${ROLLOUT_N}_r${MAX_RESPONSE_LENGTH}}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.log}"
CMD_FILE="${CMD_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.cmd.sh}"
mkdir -p "${OUTPUT_ROOT}" "${CKPT_DIR}"
export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-${OUTPUT_ROOT}/${RUN_NAME}.jsonl}"

OPTIM_ARGS=()
if [[ "${OPTIMIZER_OFFLOAD}" =~ ^(True|true|1)$ ]]; then
  OPTIM_ARGS=(
    "+actor_rollout_ref.actor.optim.override_optimizer_config.chunked_optimizer_state_offload=True"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_state_offload_chunk_size_mb=${OPTIMIZER_STATE_CHUNK_MB}"
  )
fi

VLLM="actor_rollout_ref.rollout.engine_kwargs.vllm"
BI_ROLLOUT_ARGS=()
if [[ "$BATCH_INVARIANT" == 1 ]]; then
  BI_ROLLOUT_ARGS=("+${VLLM}.attention_backend=${ATTENTION_BACKEND}" "+${VLLM}.hf_overrides={head_dtype:${DSV41_HEAD_DTYPE}}")
fi

# A pre-existing Ray cluster inherits its service environment, not this shell.
# Forward the paired kernel/compatibility contract explicitly, without secrets.
mapfile -t RUNTIME_ENV_ARGS < <(python3 -S - <<'PY'
import json, os
names = ('VLLM_BATCH_INVARIANT', 'VLLM_BI_MXFP8_GEMM', 'VLLM_BI_BF16_GEMM',
         'VLLM_BI_MXFP8_PREQUANT', 'VLLM_BI_MHC_OVERLAP', 'VLLM_BI_REPLAY_STAGE_LAYERS',
         'VERL_MLITE_SKIP_RUNTIME_PATCHES', 'VERL_MLITE_HF_CONFIG_MODEL_TYPE',
         'VERL_MLITE_ROLLOUT_SLEEP_LEVEL', 'VERL_FULL_DETERMINISM',
         'VLLM_BATCH_INVARIANT_KERNEL_LIB', 'CUBLAS_WORKSPACE_CONFIG',
         'VLLM_USE_V2_MODEL_RUNNER', 'VLLM_BI_MEGA_ATTN', 'VLLM_BI_MEGA_GATE', 'VLLM_BI_MEGA_MHC',
         'DSV41_ATTENTION_CORE', 'DSV41_HEAD_DTYPE', 'DSV41_MEGAMOE_MAX_TOKENS',
         'VERL_MLITE_VAL_MAX_TOKENS', 'VERL_MLITE_L2_RETENTION', 'VERL_MLITE_SLEEP_FP_DIR',
         'MLITE_DISTOPT_SHARDING_TYPE', 'MCORE_DIST_CKPT_STREAM_D2H', 'VERL_MLITE_ROLLOUT_ENV',
         )
for name in names:
    if name in os.environ:
        print('++ray_kwargs.ray_init.runtime_env.env_vars.' + name + '=' + json.dumps(os.environ[name]))
if os.environ.get('DS41_RAY_PY_EXECUTABLE'):
    print('++ray_kwargs.ray_init.runtime_env.py_executable=' + json.dumps(os.environ['DS41_RAY_PY_EXECUTABLE']))
PY
)

COMMAND=(
  python3 -m verl.trainer.main_ppo
  "hydra.searchpath=[pkg://verl_mlite.config]"
  # algorithm (DS4 Flash full-param DAPO)
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "algorithm.kl_ctrl.kl_coef=0.0"
  "algorithm.rollout_correction.bypass_mode=False"
  "algorithm.norm_adv_by_std_in_grpo=False"
  # data
  "data.train_files=${TRAIN_FILES}"
  "data.val_files=${VAL_FILES}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.prompt_key=prompt"
  "data.return_raw_chat=True"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "+data.apply_chat_template_kwargs.chat_template='${CHAT_TEMPLATE}'"
  "data.truncation=error"
  "data.dataloader_num_workers=8"
  # model
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.trust_remote_code=True"
  "actor_rollout_ref.model.use_fused_kernels=False"
  "actor_rollout_ref.model.custom_chat_template='${CHAT_TEMPLATE}'"
  # actor: mLite deployment operators
  "actor@actor_rollout_ref.actor=mlite_actor"
  "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE:-1e-5}"
  "actor_rollout_ref.actor.optim.weight_decay=0"
  "actor_rollout_ref.actor.optim.betas=[0.9,0.95]"
  "actor_rollout_ref.actor.optim.clip_grad=1.0"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=0"
  "actor_rollout_ref.actor.optim.lr_warmup_init=0"
  "actor_rollout_ref.actor.optim.lr_decay_style=constant"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=True"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN}"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.kl_loss_coef=0.0"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"
  "actor_rollout_ref.actor.clip_ratio_low=0.2"
  "actor_rollout_ref.actor.clip_ratio_high=0.28"
  "actor_rollout_ref.actor.clip_ratio_c=10.0"
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.engine.dtype=bfloat16"
  "actor_rollout_ref.actor.engine.model_name=deepseek_v41"
  "actor_rollout_ref.actor.engine.impl=vllm"
  "actor_rollout_ref.actor.engine.tp=1"
  "actor_rollout_ref.actor.engine.pp=${ACTOR_PP}"
  "actor_rollout_ref.actor.engine.vpp=1"
  "actor_rollout_ref.actor.engine.cp=${ACTOR_CP}"
  "actor_rollout_ref.actor.engine.ep=${ACTOR_EP}"
  "actor_rollout_ref.actor.engine.etp=1"
  "actor_rollout_ref.actor.engine.param_offload=${PARAM_OFFLOAD}"
  "actor_rollout_ref.actor.engine.optimizer_offload=${OPTIMIZER_OFFLOAD}"
  "actor_rollout_ref.actor.engine.grad_offload=${GRAD_OFFLOAD}"
  "actor_rollout_ref.actor.engine.attention_backend_override=null"
  "actor_rollout_ref.actor.engine.load_hf_weights=True"
  "actor_rollout_ref.actor.engine.export_dtype=null"
  "actor_rollout_ref.actor.engine.router_replay_mode=disabled"
  "+actor_rollout_ref.actor.engine.full_determinism=True"
  "actor_rollout_ref.actor.engine.impl_cfg.use_thd=True"
  "+actor_rollout_ref.actor.engine.impl_cfg.optimizer=adam"
  "+actor_rollout_ref.actor.engine.impl_cfg.recompute=${RECOMPUTE:-full}"
  "++actor_rollout_ref.actor.engine.impl_cfg.deterministic=True"
  # Deterministic sparse-attention backward needs a 26.55 GiB workspace at 32K (OOM).
  "+actor_rollout_ref.actor.engine.impl_cfg.attention_deterministic_backward=${ACTOR_ATTN_DETERMINISTIC_BWD:-False}"
  "+actor_rollout_ref.actor.engine.impl_cfg.attention_core=${DSV41_ATTENTION_CORE}"
  "+actor_rollout_ref.actor.engine.impl_cfg.head_dtype=${DSV41_HEAD_DTYPE}"
  # rollout: resident vLLM, BI=1 sparse arm, DEP per node
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=async"
  "actor_rollout_ref.rollout.agent.num_workers=${TOTAL_GPUS}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}"
  "actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.calculate_log_probs=True"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.logprobs_mode=raw_logprobs"
  "actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}"
  "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_SEQ_LEN}"
  "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.val_kwargs.temperature=1.0"
  "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
  "actor_rollout_ref.rollout.val_kwargs.top_k=-1"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}"
  "actor_rollout_ref.rollout.free_cache_engine=True"
  "actor_rollout_ref.rollout.load_format=auto"
  "actor_rollout_ref.rollout.enable_rollout_routing_replay=False"
  "actor_rollout_ref.rollout.enforce_eager=False"
  "actor_rollout_ref.rollout.full_determinism=${ROLLOUT_FULL_DETERMINISM}"
  # vLLM default graph mode and capture sizes (up to 768 tokens here).
  "+${VLLM}.compilation_config={mode:0}"
  "+${VLLM}.attention_config={indexer_kv_dtype:${INDEXER_KV_DTYPE},indexer_sparse_logits:${INDEXER_SPARSE_LOGITS}}"
  "+${VLLM}.kv_cache_dtype=nvfp4_ds_mla"
  "+${VLLM}.all2all_backend=allgather_reducescatter"
  "+${VLLM}.kernel_config={enable_flashinfer_autotune:true,moe_backend:deep_gemm_mega_moe}"
  "+${VLLM}.engram_config={cpu_offload:true,use_thp:true,embedding_across_dp:true,dp_shared_memory:false}"
  "+${VLLM}.language_model_only=True"
  "+${VLLM}.worker_extension_cls=verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"
  "${SPEC_ARGS[@]}"
  "${BI_ROLLOUT_ARGS[@]}"
  # trainer
  "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}"
  "critic.enable=False"
  "trainer.use_v1=False"
  "trainer.balance_batch=True"
  "trainer.logger=[console,file]"
  "trainer.project_name=verl-mlite-ds41-dapo"
  "trainer.experiment_name=${RUN_NAME}"
  "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
  "trainer.nnodes=${NNODES}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.resume_mode=${RESUME_MODE}"
  "trainer.default_local_dir=${CKPT_DIR}"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
  "trainer.log_val_generations=10"
  # reward (DAPO overlong shaping as in the DS4 recipe)
  "+reward.reward_kwargs.overlong_buffer_cfg.enable=True"
  "+reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}"
  "+reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0"
  "+reward.reward_kwargs.overlong_buffer_cfg.log=False"
  "${OPTIM_ARGS[@]}"
  "${RUNTIME_ENV_ARGS[@]}"
  "$@"
)

printf '%q ' "${COMMAND[@]}" >"${CMD_FILE}"
printf '\n' >>"${CMD_FILE}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
if [[ "${COMPOSE_ONLY:-0}" == "1" ]]; then
  "${COMMAND[@]}" --cfg job --resolve
  exit 0
fi

echo "[ds41-dapo] nodes=${NNODES} ep=${ACTOR_EP} cp=${ACTOR_CP} dep=${ROLLOUT_DP} spec=${ROLLOUT_SPEC} prompt=${PROMPT_MODE}"
set +e
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
cmd_rc="${PIPESTATUS[0]}"
set -e
exit "${cmd_rc}"
