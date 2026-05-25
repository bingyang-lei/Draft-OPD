#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Space-separated checkpoint step names, for example:
#   STEP_NAMES="global_step_5000 global_step_5500"
STEP_NAMES=${STEP_NAMES:-""}

# Template paths may contain {step_name}. Paths under this repo can be relative.
# Example:
#   ACTOR_DIR_TEMPLATE="checkpoints/verl-dflash-opd/YOUR_RUN/{step_name}/actor"
#   TARGET_DIR_TEMPLATE="checkpoints/verl-dflash-opd/YOUR_RUN/{step_name}/draft_model"
ACTOR_DIR_TEMPLATE=${ACTOR_DIR_TEMPLATE:-""} # your FSDP actor checkpoint template.
REFERENCE_DRAFT_DIR=${REFERENCE_DRAFT_DIR:-""} # your reference DFlash draft model path.
TARGET_DIR_TEMPLATE=${TARGET_DIR_TEMPLATE:-""} # your output draft model template.

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

  python scripts/extract_dflash_draft_from_fsdp.py \
    --actor-dir "${actor_dir}" \
    --reference-draft-dir "${REFERENCE_DRAFT_DIR}" \
    --target-dir "${target_dir}"
done
