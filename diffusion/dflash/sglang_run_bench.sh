#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash sglang_run_bench.sh --config-yaml PATH [--dry-run]
  bash sglang_run_bench.sh PATH [--dry-run]

The YAML file should contain a jobs list. Example:

jobs:
  - gpu: 0
    target_model: /path/to/target
    draft_model: /path/to/draft
    output_md: sglang-logs-final/result.md
    dataset_name: ["aime24:30", "math500:128"]
    concurrency_num: 1

If draft_model is empty, the job runs baseline only.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONFIG_YAML=""
DRY_RUN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-yaml|--config)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a path." >&2
        exit 1
      fi
      CONFIG_YAML="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN_ARGS=(--dry-run)
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -n "${CONFIG_YAML}" ]]; then
        echo "Error: multiple config paths provided: ${CONFIG_YAML} and $1" >&2
        exit 1
      fi
      CONFIG_YAML="$1"
      shift
      ;;
  esac
done

CONFIG_YAML=${CONFIG_YAML:-"eval_config/four_gpu_final.yaml"}

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
  source "${CONDA_ACTIVATE}" "${CONDA_ENV_NAME:-dflash-sglang-eval}"
fi

python launch_sglang_bench_jobs.py   --config-yaml "${CONFIG_YAML}"   "${DRY_RUN_ARGS[@]}"
