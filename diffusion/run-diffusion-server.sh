#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-""} # your target model path, e.g. Qwen/Qwen3-4B or a local HF model directory.
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-"${OPD_ROOT}/verl/checkpoints/verl-dflash-opd/YOUR_RUN/global_step_xxxx/draft_model"} # your OPD draft model path.
PORT=${PORT:-30000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TP_SIZE=${TP_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-fa3}

if [[ -z "${TARGET_MODEL_PATH}" ]]; then
  echo "Error: TARGET_MODEL_PATH is required." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -m sglang.launch_server \
  --model-path "${TARGET_MODEL_PATH}" \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "${DRAFT_MODEL_PATH}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --dtype bfloat16 \
  --attention-backend "${ATTENTION_BACKEND}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --trust-remote-code
