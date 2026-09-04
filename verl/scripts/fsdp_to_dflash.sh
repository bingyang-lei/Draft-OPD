#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"


# Template paths may contain {step_name}. Paths under this repo can be relative.
# Example:
#   ACTOR_DIR_TEMPLATE="checkpoints/verl-dflash-opd/YOUR_RUN/{step_name}/actor"
#   TARGET_DIR_TEMPLATE="checkpoints/verl-dflash-opd/YOUR_RUN/{step_name}/draft_model"

# /mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-8b-dflash/epoch_5_step_280000
# /mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data/epoch_5_step_295000
ACTOR_DIR_TEMPLATE=${ACTOR_DIR_TEMPLATE:-"/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/exp-qwen3-4b/reject-first_True/{step_name}/actor"} # your FSDP actor checkpoint template.
REFERENCE_DRAFT_DIR=${REFERENCE_DRAFT_DIR:-"/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data/epoch_5_step_295000"} # your reference DFlash draft model path.
TARGET_DIR_TEMPLATE=${TARGET_DIR_TEMPLATE:-"/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/exp-qwen3-4b/reject-first_True/{step_name}/draft"} # your output draft model template.

# Space-separated checkpoint step names, for example:
#   STEP_NAMES="global_step_5000 global_step_5500"
STEP_NAMES=${STEP_NAMES:-"global_step_4500 global_step_5000 global_step_5304"} # global_step_17500 global_step_20000 global_step_21216

if [[ -z "${STEP_NAMES}" || -z "${ACTOR_DIR_TEMPLATE}" || -z "${REFERENCE_DRAFT_DIR}" || -z "${TARGET_DIR_TEMPLATE}" ]]; then
  cat >&2 <<'EOF'
Error: missing required configuration.

Required environment variables:
  STEP_NAMES="global_step_5000 global_step_5500"
  ACTOR_DIR_TEMPLATE="checkpoints/verl-dflash-opd/YOUR_RUN/{step_name}/actor"
  REFERENCE_DRAFT_DIR="/path/to/reference/dflash/draft_model"
  TARGET_DIR_TEMPLATE="checkpoints/verl-dflash-opd/YOUR_RUN/{step_name}/draft_model"
EOF
  exit 1
fi

for step_name in ${STEP_NAMES}; do
  actor_dir="${ACTOR_DIR_TEMPLATE//\{step_name\}/${step_name}}"
  target_dir="${TARGET_DIR_TEMPLATE//\{step_name\}/${step_name}}"
  # echo $actor_dir
  # echo $target_dir
  # sleep 3600
  python scripts/extract_dflash_draft_from_fsdp.py \
    --actor-dir "${actor_dir}" \
    --reference-draft-dir "${REFERENCE_DRAFT_DIR}" \
    --target-dir "${target_dir}"

  rm -rf ${actor_dir}

  sleep 2
done
