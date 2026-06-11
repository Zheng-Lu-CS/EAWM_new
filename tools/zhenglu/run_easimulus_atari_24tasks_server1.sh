#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SERVER_NAME="${SERVER_NAME:-atari24_server1}"
export TASKS="${TASKS:-Alien Amidar Assault Asterix BankHeist BattleZone Breakout ChopperCommand}"

exec bash "${SCRIPT_DIR}/run_easimulus_atari_8task_4gpu_common.sh"
