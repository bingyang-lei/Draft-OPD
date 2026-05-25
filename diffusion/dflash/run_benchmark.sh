#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-""} # your target model path.
LOG_DIR=${LOG_DIR:-"logs/draft-opd-eval"}
DRAFT_MODELS_CSV=${DRAFT_MODELS_CSV:-""} # comma-separated draft model paths.
TASKS_CSV=${TASKS_CSV:-"gsm8k:128,aime24:30,math500:128,mbpp:128,humaneval:164,mt-bench:80,mgsm_zh:32"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-16}
TEMPERATURE=${TEMPERATURE:-0.0}
PRINT_CASE=${PRINT_CASE:-false}
ENABLE_THINKING=${ENABLE_THINKING:-true}

if [[ -z "${TARGET_MODEL_PATH}" ]]; then
  echo "Error: TARGET_MODEL_PATH is required." >&2
  exit 1
fi
if [[ -z "${DRAFT_MODELS_CSV}" ]]; then
  echo "Error: DRAFT_MODELS_CSV is required. Use a comma-separated list of draft model paths." >&2
  exit 1
fi

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}
export HF_HOME=${HF_HOME:-""} # your Hugging Face cache path.
if [[ -n "${HF_HOME}" ]]; then
  export HF_HUB_CACHE=${HF_HUB_CACHE:-"${HF_HOME}/hub"}
  export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"${HF_HOME}/datasets"}
fi
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_ALLOW_CODE_EVAL=${HF_ALLOW_CODE_EVAL:-1}
export FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK:-1}

if [[ -n "${CONDA_ACTIVATE:-}" ]]; then
  # Set CONDA_ACTIVATE to your conda activation script, for example /path/to/miniconda3/bin/activate.
  source "${CONDA_ACTIVATE}" "${CONDA_ENV_NAME:-python312}"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  num_gpu=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
  num_gpu=${NUM_GPU:-1}
fi

mkdir -p "${LOG_DIR}"
IFS=',' read -r -a DRAFT_MODELS <<< "${DRAFT_MODELS_CSV}"
IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"

for draft_path in "${DRAFT_MODELS[@]}"; do
  draft_tag="$(basename "$(dirname "${draft_path}")")_$(basename "${draft_path}")"
  log_file="${LOG_DIR}/${draft_tag}.log"

  CASE_ARGS=()
  THINKING_ARGS=()
  [[ "${PRINT_CASE}" == "true" ]] && CASE_ARGS=(--case)
  [[ "${ENABLE_THINKING}" == "true" ]] && THINKING_ARGS=(--enable-thinking)

  for task in "${TASKS[@]}"; do
    IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "${task}"
    torchrun       --nproc_per_node="${num_gpu}"       --master_port="${MASTER_PORT:-29600}"       ./benchmark.py       --dataset "${DATASET_NAME}"       --max-samples "${MAX_SAMPLES}"       --model-name-or-path "${TARGET_MODEL_PATH}"       --draft-name-or-path "${draft_path}"       --max-new-tokens "${MAX_NEW_TOKENS}"       --block-size "${BLOCK_SIZE}"       --temperature "${TEMPERATURE}"       --skip-base       "${CASE_ARGS[@]}"       "${THINKING_ARGS[@]}"       | tee -a "${log_file}"
  done
done
