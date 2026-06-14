#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK="${TASK:-UpNDown}"
export SERVER_NAME="${SERVER_NAME:-upndown_dense_ablation_2gpu}"
exec bash "${SCRIPT_DIR}/run_easimulus_atari_dense_ablation_2gpu_common.sh"
