# Shared configuration for the Nano-v3 MMLU-Pro disaggregation evaluation matrix.
# Every value can be overridden in the environment before invoking launch_oci.sh.

EVAL_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MEGATRON_ROOT="${MEGATRON_ROOT:-$(cd "${EVAL_CONFIG_DIR}/../../.." && pwd)}"

export EVAL_IMAGE="${EVAL_IMAGE:-/lustre/fsw/portfolios/nemotron/users/csathe/chaitrasathe+dynamo-megatron+mamba.sqsh}"
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/hf_home}"
export HF_HUB_CACHE="${EVAL_HF_HUB_CACHE:-${HF_HOME}/hub}"
export LM_EVAL_HARNESS_PATH="${LM_EVAL_HARNESS_PATH:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/lm-evaluation-harness}"

export MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/nemotron-3-nano-30b}"
export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/nanov3}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json}"
export DYNAMO_MODEL="${DYNAMO_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nemotron-3-nano-30b}"

export EVAL_OUTPUT_BASE="${EVAL_OUTPUT_BASE:-${MEGATRON_ROOT}/eval_results/nano_v3_mmlu_pro}"
export EVAL_PYTHON_DEPS_DIR="${EVAL_PYTHON_DEPS_DIR:-${EVAL_OUTPUT_BASE}/python-deps-py312}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${EVAL_OUTPUT_BASE}/hf-datasets-cache}"
export MODES="${MODES:-no_disagg,dynamo,native_nccl,native_nixl}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
# Dynamo's OpenAI frontend accepts at most 128 total choices per completion
# request. Keep the harness's original 512 default for the Megatron servers,
# but cap the Dynamo request batch separately.
export DYNAMO_BATCH_SIZE="${DYNAMO_BATCH_SIZE:-128}"
export EVAL_LIMIT="${EVAL_LIMIT:-}"
export STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-1800}"
export CLIENT_TIMEOUT_SECONDS="${CLIENT_TIMEOUT_SECONDS:-14400}"

export SERVER_PORT="${SERVER_PORT:-5000}"
export INFERENCE_MAX_SEQ_LENGTH="${INFERENCE_MAX_SEQ_LENGTH:-131072}"
export INFERENCE_BUFFER_SIZE_GB="${INFERENCE_BUFFER_SIZE_GB:-20}"
export INFERENCE_MAX_TOKENS="${INFERENCE_MAX_TOKENS:-8192}"
export INFERENCE_MAX_REQUESTS="${INFERENCE_MAX_REQUESTS:-32}"
# Thirteen exponential prefill buckets cover 8192 through 4 plus 1. The
# explicit count deliberately trims the unsupported two-token prefill bucket;
# decode graph generation still retains its two-token bucket.
export INFERENCE_NUM_CUDA_GRAPHS="${INFERENCE_NUM_CUDA_GRAPHS:-13}"
# Log every step so the matrix can reject any eager fallback from the logs.
export INFERENCE_LOGGING_STEP_INTERVAL="${INFERENCE_LOGGING_STEP_INTERVAL:-1}"
export MAMBA_PREFIX_CACHE_GB="${MAMBA_PREFIX_CACHE_GB:-4.0}"

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_pre}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export SLURM_QOS="${SLURM_QOS:-interactive}"
export SLURM_TIME="${SLURM_TIME:-4:00:00}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-1234}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
# Native NCCL overlaps point-to-point KV handoff with the shard's EP
# collectives. A single CUDA connection can serialize the two communicators
# into a scheduling cycle, so only that mode uses multiple connections.
export NATIVE_NCCL_CUDA_DEVICE_MAX_CONNECTIONS="${NATIVE_NCCL_CUDA_DEVICE_MAX_CONNECTIONS:-8}"
