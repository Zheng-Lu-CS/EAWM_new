#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to an existing EASimulus output run directory.}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT="${CHECKPOINT:-last}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_BATCHES="${NUM_BATCHES:-16}"
KEEP_RATIOS="${KEEP_RATIOS:-0.25 0.5 0.75}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/logs/decision_aware_precision_router/diagnostics/$(date +%Y%m%d_%H%M%S)}"

python "${SCRIPT_DIR}/diagnose_patch_importance.py" \
  --repo-root "${PROJECT_ROOT}" \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-batches "${NUM_BATCHES}" \
  --keep-ratios ${KEEP_RATIOS} \
  --output-dir "${OUTPUT_DIR}"

