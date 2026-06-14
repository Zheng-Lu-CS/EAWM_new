#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK="${TASK:-Freeway}"
export SERVER_NAME="${SERVER_NAME:-freeway_dense_ablation_2gpu}"
exec bash "${SCRIPT_DIR}/run_easimulus_atari_dense_ablation_2gpu_common.sh"
