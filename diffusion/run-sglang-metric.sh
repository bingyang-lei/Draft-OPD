#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BENCH_LIST=${BENCH_LIST:-mbpp}
NUM=${NUM:-10}
MAX_TOKENS=${MAX_TOKENS:-4096}
TEMPERATURE=${TEMPERATURE:-1.0}
PORTS=${PORTS:-"30000 30001"}
OUTPUT_DIR=${OUTPUT_DIR:-"./logs/quick-metric"}

mkdir -p "${OUTPUT_DIR}"

for port in ${PORTS}; do
  python sglang-metrics.py \
    --port "${port}" \
    --num "${NUM}" \
    --max-tokens "${MAX_TOKENS}" \
    --enable-thinking \
    --bench-list "${BENCH_LIST}" \
    --temperature "${TEMPERATURE}" \
    --output "${OUTPUT_DIR}/metrics-${BENCH_LIST}-${port}.log" &
done

wait
echo "eval done"
