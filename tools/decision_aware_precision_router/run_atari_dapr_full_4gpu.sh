#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_NAME="${EXP_NAME:-dapr_atari_full_keep75}"
export KEEP_RATIO="${KEEP_RATIO:-0.75}"
export TASKS="${TASKS:-Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown}"

exec bash "${SCRIPT_DIR}/run_atari_dapr_validation_4gpu.sh"

